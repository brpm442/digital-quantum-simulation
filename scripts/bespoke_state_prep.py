import numpy as np, scipy as sp, math as mt, cmath as cmt, qiskit, itertools
from qiskit import *
from generic_MPS_methods import *
from generic_state_prep import *
from useful_qc_methods import *

def GHZ_state_preparation(n, connectivity='all'):
    """
    Function that generates quantum circuit that
    prepares n-qubit GHZ state deterministically.
    Input "connectivity" can take values 'all' (in
    which case circuit depth is O(log(n))) or
    'lin' (in which case circuit depth is O(n) but
    CNOTs act only on pairs of adjacent qubits).
    Based on arXiv:1807.05572v1.
    """
    
    qc = QuantumCircuit(n)
    qc.h(0)
    
    if connectivity == 'lin':
        for i in range(n-1):
            qc.cx(i,i+1)
    else:
        for i in range(1,n):
            c = int(i - 2**(np.floor(np.log2(i))))
            qc.cx(c,i)
    
    return qc

def Neel_state_preparation(n, alpha, beta, connectivity='all'):
    """
    Function that generates a circuit that prepares a Neel 
    state of the form alpha |010101...01) + beta |101010...10) 
    for an even number of qubits n and two coefficients
    alpha and beta such that |alpha|^2 + |beta|^2 = 1.
    Connectivity is either all-to-all (i.e., connectivity='all'),
    in which case circuit depth is O(log(n)), or linear (i.e.,
    connectivity='lin'), in which case circuit depth is O(n).
    """
    if abs(np.absolute(alpha)**2 + np.absolute(beta)**2 - 1) > 1e-10:
        raise ValueError('Squares of norms of alpha and beta must sum to 1.')
        
    qc = QuantumCircuit(n)
    
    if abs(np.absolute(alpha) - 1) < 1e-10:
        for i in range(0,n,2):
            qc.x(i)
    elif abs(np.absolute(beta) - 1) < 1e-10:
        for i in range(1,n,2):
            qc.x(i)
    else:
        psi1 = [alpha, beta]
        qc1 = single_qubit_state_preparation(psi1, params_out=False).to_gate()
        qc_GHZ = GHZ_state_preparation(n, connectivity).to_gate()
        
        qc.append(qc1,[0]) # 1Q circuit to encode amplitudes alpha and beta
        qc.h(0) # Cancels Hadamard from GHZ state preparation
        qc.append(qc_GHZ, list(range(n)))
        # Applying X gates at every other qubit to obtain Neel states
        for i in range(0,n,2):
            qc.x(i)
    
    return qc

def spin_wave_state_preparation(n, k, connectivity='all'):
    """
    Function that generates a circuit that prepares a spin-wave
    state (i.e., a one-magnon state) for a n-spin-1/2 system,
    given the momentum k. Connectivity is either all-to-all 
    (i.e., connectivity='all'), in which case circuit depth is 
    O(log(n)), or linear (i.e., connectivity='lin'), in which 
    case circuit depth is O(n).
    """
    
    # Generating linear combination of basis states (i.e., W state)
    qc = W_state_preparation(n, connectivity)
    
    # Applying phase factors according to given momentum k
    for i in range(n):
        qc.rz(-k*i, i)
    
    return qc

def VBS_general_preparation_Hadamard_test(N_site, qubit_pairs, 
                                          sets_of_qubits_encoding_a_site,
                                          with_rep_mitigation=False,
                                          qc_Hadamard = None,
                                          qc_island_bulk = None):
    """
    Function that generates circuit that prepares spin-(N_site/2) VBS
    state via probabilistic method based on Hadamard test introduced in
    Sec. VI of B. Murta, P. M. Q. Cruz and J. Fernández-Rossier, Phys.
    Rev. Research 5, 013190 (2023). 
    qubit_pairs is a list of two-element lists, each containing the 
    indices of the qubit pairs forming a valence bond in the pre-VBS 
    state. qubit_pairs therefore defines the connectivity of the 
    underlying lattice. sets_of_qubits_encoding_a_site is a list of
    N_site-element lists, each corresponding to the indices of the
    qubits that define a local spin-(N_site/2) local degree of
    freedom at a lattice site.
    If with_rep_mitigation == False, every lattice site is associated
    with an ancillary qubit, which must be measured in the |1) state
    to ensure the successful symmetrization of the respective site.
    Such computational basis measurements of the ancillas are not
    included in the circuit. The ancillary qubits are the most
    significant ones and their order follows that of the respective
    sites in the sets_of_qubits_encoding_a_site input.
    If with_rep_mitigation == True, num_islands_rep_mitigation is
    called to determine which sites are symmetrized deterministically
    first (via the qc_island_bulk circuit provided as an input or, in its
    absence, generic state preparation using the SVD-based method
    by Plesch and Brukner), with the remaining sites being symmetrized
    probabilistically via the Hadamard test.
    The (N_site+1)-qubit circuit that implements the local Hadamard test
    can be provided via the qc_Hadamard input. If it is not provided, 
    then its circuit is generated via the quantum Shannon decomposition
    of the controlled-exp(-i pi Symmetrization).
    """
    
    #########################
    # Relevant subfunctions #
    #########################
    
    # Subfunction that gives ordered list of indices of qubits involved
    # in valence bonds. For example, if [2,5] is in qubit_pairs, then
    # f[2] = 5 and f[5] = 2. If qubit 10 is not involved in any valence
    # bond (i.e., it is at the boundary), f[10] = 10.
    def valence_bond_connection_function(qu_pairs, num_qubits):
        f = list(range(num_qubits))
        for i in qu_pairs:
            f[i[0]] = i[1]
            f[i[1]] = i[0]
        return f

    # Subfunction that determines the position in the list
    # qubits_for_sites that corresponds to the site to which 
    # the qubit with the given index belongs.
    def site_finder_given_qubit_index(index, qubits_for_sites):
        for i in range(len(qubits_for_sites)):
            if index in qubits_for_sites[i]:
                return i

    # Subfunction that determines how many islands can be prepared
    # deterministically in first part (if with_rep_mitigation == True),
    # which valence bonds still have to be initialized explicitly, and
    # which sites must be symmetrized probabilistically in second part
    def num_islands_rep_mitigation(q_pairs, n_qubits, 
                                   qubits_encoding_a_site):
        
        first_sites = []
        second_sites = []
        remaining_bonds = q_pairs.copy()
        sets_of_qubits_encoding_a_site_copy = qubits_encoding_a_site.copy()
        f = valence_bond_connection_function(q_pairs, n_qubits)
        
        while len(sets_of_qubits_encoding_a_site_copy) > 0:
            new_island = sets_of_qubits_encoding_a_site_copy[0]
            first_sites.append(new_island)
            del sets_of_qubits_encoding_a_site_copy[0]
            for j in range(len(new_island)):
                qubit_index = f[new_island[j]]
                site = site_finder_given_qubit_index(qubit_index, sets_of_qubits_encoding_a_site_copy)
                if site is not None:
                    second_sites.append(sets_of_qubits_encoding_a_site_copy[site])
                    del sets_of_qubits_encoding_a_site_copy[site]
            for i in range(len(new_island)):
                for j in range(len(remaining_bonds)):
                    if new_island[i] in remaining_bonds[j]:
                        del remaining_bonds[j]
                        break
        
        return first_sites, second_sites, remaining_bonds

    # Subfunction that generates statevector corresponding to an island
    # that has no qubits at the boundary (i.e., all qubits are involved
    # in a valence bond in the pre-VBS state before symmetrization). The
    # N_site qubits encoding the central site of the island are the most
    # significant ones. The remaining N_sites qubits encode other sites.
    def island_statevector_bulk(N_site):
        valence_bond = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
        pre_VBS = valence_bond
        for i in range(N_site-1):
            pre_VBS = np.kron(pre_VBS, valence_bond)
        pre_VBS = np.reshape(pre_VBS, (2,)*(2*N_site))
        new_order = list(range(0,2*N_site,2))+list(range(1,2*N_site,2))
        pre_VBS = np.transpose(pre_VBS, new_order)
        pre_VBS = pre_VBS.flatten()
        
        Symm = symmetrization(N_site)
        Symm = np.kron(Symm, np.eye(Symm.shape[0]))
        VBS_island = Symm.dot(pre_VBS)
        VBS_island = 1/np.linalg.norm(VBS_island)*VBS_island
        
        VBS_island = np.reshape(VBS_island, (2,)*(2*N_site))
        VBS_island = np.transpose(VBS_island, np.argsort(new_order))
        VBS_island = VBS_island.flatten()
        
        return VBS_island 

    # Subfunction that generates statevector corresponding to an island
    # that has at least one qubit at the boundary (i.e., one qubit is not
    # involved in a valence bond in the pre-VBS state before symmetrization). 
    # The N_site qubits encoding the central site of the island are the most
    # significant ones. The remaining N_sites qubits encode other sites.
    def island_statevector_boundary(boundary_array):
        valence_bond = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
        isolated_spin = np.array([1,0])
        num_qubits = len(boundary_array)
        
        if boundary_array[0] == 1:
            pre_VBS = isolated_spin
            central_qubits_indices = [0]
            other_qubits_indices = []
            i = 1
        else:
            pre_VBS = valence_bond
            other_qubits_indices = [0]
            central_qubits_indices = [1]
            i = 2
        
        while i < len(boundary_array):
            if boundary_array[i] == 1:
                pre_VBS = np.kron(pre_VBS, isolated_spin)
                central_qubits_indices.append(i)
                i = i + 1
            else:
                pre_VBS = np.kron(pre_VBS, valence_bond)
                other_qubits_indices.append(i)
                central_qubits_indices.append(i+1)
                i = i + 2
                
        pre_VBS = np.reshape(pre_VBS, (2,)*(num_qubits))
        new_order = central_qubits_indices + other_qubits_indices
        pre_VBS = np.transpose(pre_VBS, new_order)
        pre_VBS = pre_VBS.flatten()
        
        Symm = symmetrization(N_site)
        Symm = np.kron(Symm, np.eye(int(2**(num_qubits)/Symm.shape[0])))
        VBS_island = Symm.dot(pre_VBS)
        VBS_island = 1/np.linalg.norm(VBS_island)*VBS_island
        
        VBS_island = np.reshape(VBS_island, (2,)*(num_qubits))
        VBS_island = np.transpose(VBS_island, np.argsort(new_order))
        VBS_island = VBS_island.flatten()
        
        return VBS_island          
    
    #################
    # Main function #
    #################
    
    num_sites = len(sets_of_qubits_encoding_a_site)
    num_qubits_main_register = N_site*num_sites
    
    # Generating circuit for controlled-exp(-i pi Symmetrization)
    # via quantum Shannon decomposition if qc_Hadamard was not provided
    if qc_Hadamard is None:
        Symm = symmetrization(N_site)
        U = sp.linalg.expm(-1j*np.pi*Symm)
        cU = direct_sum(np.eye(U.shape[0]), U)
        cU_circuit = qiskit.synthesis.unitary.qsd.qs_decomposition(cU)
        cU_gate = cU_circuit.to_gate()
        
        qc_Hadamard = QuantumCircuit(N_site+1)
        qc_Hadamard.h(N_site)
        qc_Hadamard.append(cU_gate, list(range(N_site+1)))
        qc_Hadamard.h(N_site)
    
    if with_rep_mitigation:
        first_sites, second_sites, remaining_bonds = num_islands_rep_mitigation(qubit_pairs, 
                                                                                num_qubits_main_register, 
                                                                                sets_of_qubits_encoding_a_site)
        f = valence_bond_connection_function(qubit_pairs, num_qubits_main_register)
        num_ancillas = len(second_sites)
        total_num_qubits = num_qubits_main_register + num_ancillas
        qc = QuantumCircuit(total_num_qubits)
        
        # Generating circuit that prepares an island without any 
        # boundary qubits if not provided as input
        if qc_island_bulk is None:
            island_bulk = island_statevector_bulk(N_site)
            qc_island_bulk = Plesch_Brukner_preparation(island_bulk, 2*N_site, all_parts=False).to_gate()
        
        # Preparing islands deterministically via generic state preparation method
        for i in range(len(first_sites)):
            island_indices = []
            boundary_flags = []
            for j in range(N_site):
                if f[first_sites[i][j]] == first_sites[i][j]:
                    boundary_flags.append(1)
                    island_indices.append(first_sites[i][j])
                else:
                    boundary_flags.append(0)
                    boundary_flags.append(0)
                    island_indices.append(f[first_sites[i][j]])
                    island_indices.append(first_sites[i][j])
            
            if all(x == 0 for x in boundary_flags):
                qc.append(qc_island_bulk, island_indices)
            else:
                island_boundary = island_statevector_boundary(boundary_flags)
                qc_island_boundary = Plesch_Brukner_preparation(island_boundary, len(boundary_flags), all_parts=False).to_gate()
                qc.append(qc_island_boundary, island_indices[::-1])
            
        # Initializing valence bonds that have not been included in islands
        for i in remaining_bonds:
            qc.h(i[0])
            qc.x(i[1])
            qc.cx(i[0],i[1])
            qc.z(i[0])
        
        # Preparing remaining sites probabilistically via Hadamard test
        for i in range(len(second_sites)):
            ancilla_index = num_qubits_main_register+i
            qc.append(qc_Hadamard, second_sites[i]+[ancilla_index])
        
    else:
        num_ancillas = num_sites
        total_num_qubits = num_qubits_main_register + num_ancillas
        qc = QuantumCircuit(total_num_qubits)
    
        # Preparing valence bonds
        for i in qubit_pairs:
            qc.h(i[0])
            qc.x(i[1])
            qc.cx(i[0],i[1])
            qc.z(i[0])
        
        # Hadamard test at every site
        for i in range(len(sets_of_qubits_encoding_a_site)):
            ancilla_index = num_qubits_main_register+i
            qc.append(qc_Hadamard, sets_of_qubits_encoding_a_site[i]+[ancilla_index])
    
    return qc
    

def VBS_spin_1_preparation_Yale_sequential_end(n, BC, RHS_BC=0, 
                                               with_SWAP_test=False):
    """
    Function that generates quantum circuit to prepare spin-1 VBS
    state, the ground state of the 1D AKLT model. Specifically, the
    standard n-depth sequential method discussed in Sec. III.A of
    K. C. Smith, E. Crane, N. Wiebe and S. M. Girvin, PRX Quantum
    4, 020315 (2023) [see, in particular, Fig. 1(b) and Eq. 6] is
    implemented, where n is the number of sites (i.e., spins-1).
    
    For BC == 'OBC' (open boundary conditions), this method is 
    identical to the generic MPS method implemented in function 
    MPS_preparation_quantum_circuit_OBC, except that boundary 
    conditions are imposed explicitly on quantum hardware, namely
    through the addition of an ancillary qubit. The RHS boundary 
    condition is provided via the input RHS_BC, which can be 
    either 0 (spin-up) or 1 (spin-down). The RHS BC is imposed 
    right at the start of the circuit, but the LHS BC results 
    from the measurement of the ancilla at the end in the 
    computational basis, yielding either outcome (spin-up or 
    spin-down) with 50% of probability. This measurement is not
    included explicitly in the quantum circuit, as further 
    coherent manipulations may be considered.
    
    For BC == 'PBC' (periodic boundary conditions):
        
        If with_SWAP_test == False, 2 ancillas are used, which 
        correspond to the two most significant qubits at the end 
        of the circuit. Projecting these two qubits onto the spin 
        singlet 1/sqrt(2) [|01) - |10)] ensures the VBS state with 
        PBCs is successfully prepared in the remaining 2n qubits. 
        Transformation from computational basis to Bell basis is 
        already included in circuit, so the only part that is left
        as post-processing is the actual measurement in computational
        basis, just in case further coherent manipulation of the main
        2n-qubit register is required. Measuring ancillas in |00)
        occurs with 25% probability and ensures successful preparation.
    
        If with_SWAP_test == True, 3 ancillas are used, again the
        most significant qubits (2n, 2n+1, 2n+2). In these qubits
        the SWAP test is implemented at the end of the circuit, in
        line with the discussion in Appendix A of the reference.
        Measuring the most significant qubit 2n+2 in |1) ensures
        successful preparation of n-site VBS state with PBCs
        on the main 2n-qubit register, with the other two ancillas
        (2n, 2n+1) separated from those 2n qubits and encoding a 
        spin singlet. As in the option with_SWAP_test == False, 
        this happens with 25% of probability. 
        If instead the most significant qubit is measured in |0), 
        symmetrization is unsuccessful, but the resulting defect can 
        be corrected by applying a Y gate at the other two ancillas 
        (2n, 2n+1), in which case we obtain a (n+1)-site VBS state 
        [notice the extra site] with PBCs on 2n+2 qubits. That is,
        the ancillas (2n, 2n+1) become part of the main register 
        and encode a site of the VBS state. Hence, unlike the 
        previous approach, this method is deterministic even for 
        periodic boundary conditions.
    """
    
    # Generating 3-qubit unitary U
    MPS_VBS = MPS_canonical_examples('VBS-PBC-qubits', n)
    local_tensor = MPS_VBS[0].reshape((8,2))
    C = sp.linalg.null_space(np.conjugate(np.transpose(local_tensor)))
    U = np.zeros((8,8), dtype=complex)
    U[:,:2] = local_tensor
    U[:,2:] = C
    U_gate = qiskit.quantum_info.operators.Operator(U)
    
    # Generating quantum circuit    
    if BC == 'OBC':
        qc = QuantumCircuit(2*n+1)
        # Imposing boundary conditions on RHS (i.e., spin-up or spin-down)
        if RHS_BC == 1:
            qc.x(0)
        for i in range(n):
            qc.unitary(U_gate, list(range(2*i,2*i+3)))
    else:
        if with_SWAP_test:
            qc = QuantumCircuit(2*n+3)
            # Initializing spin singlet at qubits 0 and 2n+1
            qc.h(2*n+1)
            qc.x(0)
            qc.cx(2*n+1,0)
            qc.z(2*n+1)
            for i in range(n):
                qc.unitary(U_gate, list(range(2*i,2*i+3)))
            # Applying SWAP test at qubits 2n and 2n+1 with ancilla at 2n+2
            qc.h(2*n+2)
            qc.cswap(2*n+2,2*n+1,2*n)
            qc.h(2*n+2)
            qc.x(2*n+2)
        else:
            qc = QuantumCircuit(2*n+2)
            # Initializing spin singlet at qubits 0 and 2n+1
            qc.h(2*n+1)
            qc.x(0)
            qc.cx(2*n+1,0)
            qc.z(2*n+1)
            for i in range(n):
                qc.unitary(U_gate, list(range(2*i,2*i+3)))
            # Changing from computational basis to Bell state basis for (2n,2n+1)
            qc.z(2*n+1)
            qc.cx(2*n+1,2*n)
            qc.x(2*n)
            qc.h(2*n+1)
    
    return qc    

def VBS_spin_1_preparation_Yale_sequential_middle(n, BC, with_SWAP_test=False):
    """
    Function that generates quantum circuit to prepare spin-1 VBS
    state, the ground state of the 1D AKLT model. Specifically, the
    standard (n/2)-depth sequential method [starting in the middle
    of the chain or ring] discussed in Sec. III.B of K. C. Smith, 
    E. Crane, N. Wiebe and S. M. Girvin, PRX Quantum 4, 020315 
    (2023) [see, in particular, Fig. 2 and Eq. 8] is implemented, 
    where n is the number of sites (i.e., spins-1).
    
    For BC == 'OBC' (open boundary conditions), the qubits that set
    the boundaries at the least and most significant ones (0, 2n+1).
    Both have to be measured in the computational basis to impose
    the boundary conditions. All four combinations of BCs (up-up,
    up-down, down-up, down-down) have identical probability of
    occurrence, ignoring finite-size effects. The resulting VBS
    state with OBCs is therefore prepared in the remaining 2n qubits,
    corresponding to a n-site chain.
    
    For BC == 'PBC' (periodic boundary conditions):
        
        If with_SWAP_test == False, 2 ancillas are used. They 
        correspond to qubits 0 and 2n+1 at the end of the circuit,
        just like for OBCs. Projecting these two qubits onto the spin 
        singlet 1/sqrt(2) [|01) - |10)] ensures the VBS state with 
        PBCs is successfully prepared in the remaining 2n qubits. 
        Transformation from computational basis to Bell basis is 
        already included in circuit, so the only part that is left
        as post-processing is the actual measurement in computational
        basis, just in case further coherent manipulation of the main
        2n-qubit register is required. Measuring ancillas in |00)
        occurs with 25% probability and ensures successful preparation.
    
        If with_SWAP_test == True, 3 ancillas are used, at positions
        (0, 2n+1, 2n+2) at the end of the circuit. In these qubits
        the SWAP test is implemented, in line with the discussion 
        in Appendix A of the reference. Measuring the most significant 
        qubit 2n+2 in |1) ensures successful preparation of n-site 
        VBS state with PBCs on the main 2n-qubit register (qubits 1,2,
        ..., 2n-1, 2n), with the other two ancillas (0, 2n+1) separated 
        from those 2n qubits and encoding a spin singlet. As in the 
        option with_SWAP_test == False, the probability is of 25%. 
        If instead the most significant qubit is measured in |0), 
        symmetrization is unsuccessful, but the resulting defect can 
        be corrected by applying a Y gate at the other two ancillas 
        (0, 2n+1), in which case we obtain a (n+1)-site VBS state 
        [notice the extra site] with PBCs on 2n+2 qubits. That is,
        the ancillas (0, 2n+1) become part of the main register 
        and encode a site of the VBS state. In this case, to make
        sure that all pairs of qubits a site are next to each other
        in a linear architecture, a permutation of the qubits is
        implemented to move qubit 0 past the 2n qubits in the
        original main register to be next to qubit 2n+1.
    """
    
    # Generating 3-qubit unitary U
    MPS_VBS = MPS_canonical_examples('VBS-PBC-qubits', n)
    local_tensor = MPS_VBS[0].reshape((8,2))
    C = sp.linalg.null_space(np.conjugate(np.transpose(local_tensor)))
    U = np.zeros((8,8), dtype=complex)
    U[:,:2] = local_tensor
    U[:,2:] = C
    U_gate = qiskit.quantum_info.operators.Operator(U)
    
    if with_SWAP_test:
        qc = QuantumCircuit(2*n+3)
    else:
        qc = QuantumCircuit(2*n+2)
    
    # Initializing spin singlet at qubits n and n+1 at center of lattice
    qc.h(n+(1-n%2))
    qc.x(n-n%2)
    qc.cx(n+(1-n%2),n-n%2)
    qc.z(n+(1-n%2))
    
    for i in range((n-n%2)//2):
        qc.unitary(U_gate, list(reversed(range(n+(1-n%2)-2*i-3,n+(1-n%2)-2*i))))
    for i in range((n+n%2)//2):
        qc.unitary(U_gate, list(range(n+(1-n%2)+2*i,n+(1-n%2)+2*i+3)))
        
    # Permuting qubits to move qubit 0 to position 2n
    permutation_list = [2*n] + list(range(2*n))
    qc_perm = permutation_qcircuit(permutation_list, connectivity='all').to_gate()
    qc.append(qc_perm, list(range(2*n+1)))
    
    if BC == 'PBC':
        if with_SWAP_test:
            # Applying SWAP test at qubits 0 and 2n+1 with ancilla at 2n+2
            qc.h(2*n+2)
            qc.cswap(2*n+2,2*n+1,2*n)
            qc.h(2*n+2)
            qc.x(2*n+2)
        else:
            # Changing from computational basis to Bell state basis for (0,2n+1)
            qc.z(2*n+1)
            qc.cx(2*n+1,2*n)
            qc.x(2*n)
            qc.h(2*n+1)
    else:
        # Making sure that boundary conditions match comp. basis outcomes
        qc.x(2*n)

    return qc     

def VBS_spin_1_preparation_Yale_constant_depth(n, BC, with_SWAP_test=False):
    """
    Function that generates quantum circuit to prepare spin-1 VBS
    state, the ground state of the 1D AKLT model. Specifically, the
    state-of-the-art constant-depth deterministic method discussed 
    in Sec. III.C of K. C. Smith, E. Crane, N. Wiebe and S. M. Girvin, 
    PRX Quantum 4, 020315 (2023) [see, in particular, Figs. 3, 4 and 
    Table I] is implemented, where n is the number of sites (i.e., spins-1).
    
    The key difference between this method and the previous ones is that
    one considers the preparation of dimers in parallel and then fuses
    them through measurements of ancillas in the Bell basis. If the singlet
    is obtained, then the fusion is successful, otherwise defects are added.
    The correction of these defect via post-processing does not commute with
    the operations associated with the imposition of the overall boundary
    conditions (i.e., Bell basis measurement or SWAP test for PBCs, and 
    computational basis measurements for OBCs). As a result, unlike in
    the implementations of the previous methods, the subcircuits associated
    with the SWAP test or the basis changes in the outermost ancillas are
    not included in the circuit that is returned by this function. They
    must therefore be applied explicitly in the post-processing part after
    having corrected the defects.
    """
    
    # Generating 3-qubit unitary U
    MPS_VBS = MPS_canonical_examples('VBS-PBC-qubits', n)
    local_tensor = MPS_VBS[0].reshape((8,2))
    C = sp.linalg.null_space(np.conjugate(np.transpose(local_tensor)))
    U = np.zeros((8,8), dtype=complex)
    U[:,:2] = local_tensor
    U[:,2:] = C
    U_gate = qiskit.quantum_info.operators.Operator(U)
    
    if with_SWAP_test:
        qc = QuantumCircuit(3*n-n%2+1)
    else:
        qc = QuantumCircuit(3*n-n%2)
    
    # Considering every dimer of adjacent spins-1, which
    # involves 6 qubits (4 memory qubits and 2 ancillas)
    for i in range(n//2-n%2):
        # Initializing singlet at central qubits of dimer (6i+2, 6i+3)
        qc.h(6*i+2)
        qc.x(6*i+3)
        qc.cx(6*i+2,6*i+3)
        qc.z(6*i+2)
        
        # Applying U gate on either side
        qc.unitary(U_gate, [6*i+2, 6*i+1, 6*i])
        qc.unitary(U_gate, [6*i+3, 6*i+4, 6*i+5])
    
    # Adding a trimer at the end for an odd number of sites,
    # which involves 8 qubits (6 memory qubits and 2 ancillas)
    if n%2 == 1:
        # Leftmost qubit of trimer
        qref = 3*(n-3)
        # Initializing singlet at qubits (qref+2,qref+3)
        qc.h(qref+2)
        qc.x(qref+3)
        qc.cx(qref+2,qref+3)
        qc.z(qref+2)
        
        # Applying U gate once on LHS and twice on RHS
        qc.unitary(U_gate, [qref+2, qref+1, qref])
        qc.unitary(U_gate, [qref+3, qref+4, qref+5])
        qc.unitary(U_gate, [qref+5, qref+6, qref+7])
        
    # Permuting qubits to have all ancillas as most significant
    permutation_list = []
    for i in range(n//2-n%2):
        permutation_list = permutation_list + [6*i+1,6*i+2,6*i+3,6*i+4]
    if n%2 == 1:
        permutation_list = permutation_list + list(range(qref+1,qref+7))
    permutation_list = permutation_list + [0,3*n-1-n%2]
    for i in range(n//2-1):
        permutation_list = permutation_list + [6*i+5,6*(i+1)]
    permutation_list = np.argsort(permutation_list)
    qc_perm = permutation_qcircuit(permutation_list, connectivity='all').to_gate()
    qc.append(qc_perm, list(range(3*n-n%2)))
    
    # Changing from computational basis to Bell basis for all ancilla pairs in bulk
    for i in range(1,n//2):
        qc.cx(2*n+2*i,2*n+2*i+1)
        qc.h(2*n+2*i)

    return qc     

def W_state_preparation(n, connectivity='all'):
    """
    Function that generates quantum circuit that
    prepares n-qubit W state (i.e., Dicke state with
    Hamming weight 1) deterministically. Input 
    "connectivity" can take values 'all' (in
    which case circuit depth is O(log(n))) or
    'lin' (in which case circuit depth is O(n) but
    CNOTs act only on pairs of adjacent qubits).
    Based on arXiv:1807.05572v1, but with a difference:
    for connectivity='all' (where a dichotomy tree is
    required to generate the right sequence of operations
    to yield a log-depth circuit), instead of swapping 
    pairs of children with one leaf (1,1), whenever 
    ratio p of leaf is greater than 1/2 (e.g., p = 2/3 
    for leaf (2,3)), p is replaced with 1-p so that the 
    largest factor always goes to the lower leaf.
    """
    
    # subfunction that generates dichotomy tree
    def dichotomy_tree_generator(n):
        first_generation = [[np.floor(n/2),n]]
        dichotomy_tree = [first_generation]
        current_generation = first_generation
        growing_branches_counter = 1
        
        while growing_branches_counter > 0:
            growing_branches_counter = 0
            next_generation = []
            
            for leaf in current_generation:
                if leaf not in [[0,1], [1,1], [1,2]]:
                    [n,m] = leaf
                    child_1 = [np.floor(n/2), np.floor(m/2)]
                    child_2 = [np.ceil(n/2), np.ceil(m/2)]
                    growing_branches_counter = growing_branches_counter + 1
                else:
                    child_1 = [0,1]
                    child_2 = [0,1]
                next_generation.append(child_1)
                next_generation.append(child_2)
            if growing_branches_counter > 0:
                dichotomy_tree.append(next_generation)
                current_generation = next_generation
        
        return dichotomy_tree
    
    # subfunction that generates tree of qubit pairs
    def qubits_tree_generator(dichotomy_tree):
        num_generations = len(dichotomy_tree)
        qubits_tree = [[0,1]]
        
        for gen_i in range(1,num_generations):
            targets_list = list(range(int(2**gen_i), int(2**(gen_i+1))))
            new_qubits_generation = []
            for tar_i in range(len(targets_list)):
                new_qubit_pair = [qubits_tree[gen_i-1][tar_i],targets_list[tar_i]]
                new_qubits_generation.append(new_qubit_pair[0])
                new_qubits_generation.append(new_qubit_pair[1])
            qubits_tree.append(new_qubits_generation)
        
        return qubits_tree
    
    qc = QuantumCircuit(n)
    qc.x(0)
    
    if connectivity == 'lin':
        for i in range(n-1):
            p = 1/(n-i)
            theta = np.arctan(np.sqrt(p/(1-p)))
            qc.u(theta,0,0,i+1)
            qc.cx(i,i+1)
            qc.u(-theta,0,0,i+1)
            qc.cx(i+1,i)
    else:
        dichotomy_tree = dichotomy_tree_generator(n)
        qubits_tree = qubits_tree_generator(dichotomy_tree)
            
        # Generating quantum circuit
        for i in range(len(dichotomy_tree)):
            nonzero_counter = 0
            for j in range(len(dichotomy_tree[i])):
                ratio = dichotomy_tree[i][j][0] / dichotomy_tree[i][j][1]
                if ratio > 1/2:
                    p = 1 - ratio
                else:
                    p = ratio
                if p > 1e-10 and p < 1 - 1e-10:
                    control = qubits_tree[i][2*j]
                    target = qubits_tree[i][2*nonzero_counter+1]
                    theta = np.arctan(np.sqrt(p/(1-p)))
                    qc.u(theta,0,0,target)
                    qc.cx(control,target)
                    qc.u(-theta,0,0,target)
                    qc.cx(target,control)
                    nonzero_counter = nonzero_counter + 1
    return qc
        

