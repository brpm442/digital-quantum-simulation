import numpy as np, scipy as sp, math as mt, cmath as cmt
from qiskit import *
from useful_maths_methods import *
from useful_computing_methods import *

def fermionic_version_of_spin_wave_function(spin_wf_qc, N, order='spin', 
                                            Sz_good=True, connectivity='all'):
    """
    Function that generates quantum circuit that converts a multi-spin-1/2
    wave function into its fermionic version at half-filling. Follows Sec.
    III of B. Murta and J. Fernández-Rossier, Phys. Rev. B 109, 035128 (2024).
    Jordan-Wigner transformation is assumed for fermion-to-qubit mapping.
    spin_wf_qc is a quantum circuit that prepares the N-spin-1/2 wave function.
    If order == 'spin', qubits are ordered by spin, i.e., spin-down orbitals
    in the N least significant qubits and spin-up orbitals in the N most significant
    ones, where the mapping {|0) - spin-up, |1) - spin-down} is assumed in the
    spin wave function. If order == 'site', qubits are ordered by site, i.e.,
    even qubits correspond to spin-down orbitals and odd qubits to spin-up orbitals.
    If sz_good == True and connectivity='all', constant-depth circuits are generated 
    for both orders, otherwise O(N)-depth overhead applied to order='spin'.
    If connectivity='lin', networks of fermionic SWAPs are added to make sure two-qubit
    gates are only applied between pairs of adjacent qubits.
    """

    # Subfunction that determines positions of pairs of qubits on which
    # fermionic SWAPs are applied to change order from site to spin 
    # (if direction == 'site2spin') or from spin to site (if direction
    # == 'spin2site'). N is the number of lattice sites.
    def fSWAP_network_qubit_pairs(N, direction='site2spin'):
        layers_of_fSWAPs = []
        for i in range(N-1):
            new_layer_of_fSWAPs = []
            for j in range(N-i-1):
                new_qubit_pair = [i+2*j+1,i+2*j+2]
                new_layer_of_fSWAPs.append(new_qubit_pair)
            layers_of_fSWAPs.append(new_layer_of_fSWAPs)

        if direction == 'site2spin':
            return layers_of_fSWAPs
        else:
            layers_of_fSWAPs.reverse()
            return layers_of_fSWAPs

    # If Sz_good == False and order == 'spin' we must move from spin order 
    # to site order and back because constant-depth method does not work in that case.
    if not Sz_good and order == 'spin':
        connectivity = 'lin'
    
    q = QuantumRegister(2*N)
    qc = QuantumCircuit(q)

    # All-to-all connectivity
    if connectivity == 'all':
        # All-to-all connectivity, qubits ordered by spin
        if order == 'spin':
            # Initialization of spin wave function in first N qubits
            spin_qubits = []
            for i in range(N):
                spin_qubits.append(q[i])
            state_preparation_subcirc = spin_wf_qc.to_instruction()
            qc.append(state_preparation_subcirc, spin_qubits)

            # Sitewise application of spin-to-fermion mapping 
            # (valid for conserved Sz only)
            for i in range(0,N,2):
                qc.x(q[i])
                qc.cx(q[i],q[N+i])
                qc.x(q[i])
            for i in range(1,N,2):
                qc.x(q[N+i])
                qc.x(q[i])
                qc.h(q[N+i])
                qc.cx(q[i],q[N+i])
                qc.h(q[N+i])
                qc.x(q[N+i])
                qc.cx(q[i],q[N+i])
                qc.x(q[i])
                
        # All-to-all connectivity, qubits ordered by site
        else:
            # Initializing spin wave function in even qubits only
            spin_qubits = []
            for i in range(N):
                spin_qubits.append(q[2*i])
            state_preparation_subcirc = spin_wf_qc.to_instruction()
            qc.append(state_preparation_subcirc, spin_qubits)

            # Sitewise application of spin-to-fermion mapping
            for i in range(N):
                qc.x(q[2*i])
                qc.cx(q[2*i],q[2*i+1])
                qc.x(q[2*i])
                
    # Linear connectivity
    else:
        # Initialization of spin wave function in first N qubits 
        # (for both spin and site order)
        spin_qubits = []
        for i in range(N):
            spin_qubits.append(q[i])
        state_preparation_subcirc = spin_wf_qc.to_instruction()
        qc.append(state_preparation_subcirc, spin_qubits)

        # Network of fermionic SWAPs (for both spin and site order)
        qubit_pairs_fSWAPs = fSWAP_network_qubit_pairs(N, direction='spin2site')
        for i in range(len(qubit_pairs_fSWAPs)):
            for j in range(len(qubit_pairs_fSWAPs[i])):
                qubit_1 = qubit_pairs_fSWAPs[i][j][0]
                qubit_2 = qubit_pairs_fSWAPs[i][j][1]
                # fSWAP
                qc.h(q[qubit_1])
                qc.cx(q[qubit_1],q[qubit_2])
                qc.cx(q[qubit_2],q[qubit_1])
                qc.h(q[qubit_2])

        # Single-site spin-to-fermion mapping (for both spin and site order)
        for i in range(N):
            qc.x(q[2*i])
            qc.cx(q[2*i],q[2*i+1])
            qc.x(q[2*i])

        # Network of fermionic SWAPs to return to spin order
        if order == 'spin':
            qubit_pairs_fSWAPs = fSWAP_network_qubit_pairs(N, direction='site2spin')
            for i in range(len(qubit_pairs_fSWAPs)):
                for j in range(len(qubit_pairs_fSWAPs[i])):
                    qubit_1 = qubit_pairs_fSWAPs[i][j][0]
                    qubit_2 = qubit_pairs_fSWAPs[i][j][1]
                    # fSWAP
                    qc.h(q[qubit_1])
                    qc.cx(q[qubit_1],q[qubit_2])
                    qc.cx(q[qubit_2],q[qubit_1])
                    qc.h(q[qubit_2])
        
    backend = Aer.StatevectorSimulator()
    qc_t = transpile(qc, backend)
    job = backend.run(qc_t).result()
    outputstate = job.data()['statevector'].data
    
    return outputstate, qc

def long_range_CNOT_lin_con(n, order='ct'):
    """
    Function that outputs circuit that realizes CNOT between non-adjacent
    qubits separated by n qubits in architecture restricted to linear 
    connectivity. Resulting circuit has width n+2, CNOT count ~4n and 
    depth ~n. The two qubits on which the CNOT acts nontrivially are the
    outermost ones. Control-qubit is the top one (i.e., the most significant) 
    and target-qubit is the bottom one (i.e., the least significant). If one 
    wishes the reverse order, just set order = 'tc' instead of the default 
    order = 'ct', which will add a pair of Hadamard gates on either side of
    the circuit. Follows P. M. Q. Cruz and B. Murta, APL Quantum 1, 016105
    (2024). See, in particular, Sec. III and Fig. 4(c).
    """
    
    qc = QuantumCircuit(n+2)

    m = n//2 + 1
    
    if order == 'tc':
        qc.h(0)
        qc.h(n+1)
    
    for i in range(m-1):
        qc.cx(i,i+1)
    for i in range(m-1):
        qc.cx(i+1,i)
    for i in range(n-m):
        qc.cx(n-i,n+1-i)
    for i in range(n-m):
        qc.cx(n+1-i,n-i)
    
    qc.cx(m,m-1)
    qc.cx(m+1,m)
    qc.cx(m,m-1)
    qc.cx(m+1,m)
    
    for i in reversed(range(n-m)):
        qc.cx(n+1-i,n-i)
    for i in reversed(range(n-m)):
        qc.cx(n-i,n+1-i)
    for i in reversed(range(m-1)):
        qc.cx(i+1,i)
    for i in reversed(range(m-1)):
        qc.cx(i,i+1)
        
    if order == 'tc':
        qc.h(0)
        qc.h(n+1)
            
    return qc

def long_range_Fredkin_lin_con(n1, n2, order='ctt', min_metric = 'depth'):
    """
    Function that outputs circuit that realizes Fredkin gate on trio of
    non-adjacent qubits in architecture restricted to linear connectivity.
    Specifically, one target-qubit is in position 0, separated by n1 qubits
    from the other target-qubit (at position n1+1), which is separated
    by n2 qubits from the control-qubit (at the most significant position
    n1+n2+2). This corresponds to the default order = 'ctt'. If control-qubit 
    is meant to be at the central position, order = 'tct'. If control-qubit is 
    meant to be at the least significant position, order = 'ttc'. 
    Follows P. M. Q. Cruz and B. Murta, APL Quantum 1, 016105 (2024). See, in 
    particular, Sec. III and Fig. 5.
    """
    
    qc = QuantumCircuit(n1+n2+3)

    # Subfunction that determines optimal intermediate positions,
    # i.e., after qubit rerouting, Fredkin gate is applied at trio
    # of adjacent qubits (m-1,m,m+1)
    def optimal_m_Fredkin(n1, n2, min_metric):
        
        def cnot_swapping_gate_count(s):
            if s == 1:
                return 2
            else:
                return 2+s
        
        num_hops_array = []
        depth_array = []
        for m in range(1,n2+n1+2):
            delta_c = n2 + n1 + 2 - (m + 1)
            delta_t1 = abs(n1 + 1 - m)
            delta_t2 = m - 1
            
            if n1 == 0:
                num_hops = 2*delta_c + 2*delta_t1 + 2*delta_t2
            else:
                num_hops = 2*delta_c + 3*delta_t1 + 3*delta_t2
            
            if m < n1 + 1: 
                if n2 == 0:
                    depth = max(cnot_swapping_gate_count(delta_c) + 3, 3*delta_t2)
                else:
                    depth = max(cnot_swapping_gate_count(delta_c), 3*delta_t2)
            else:
                if n1 == 0:
                    depth = max(cnot_swapping_gate_count(delta_c), cnot_swapping_gate_count(delta_t2) + 2)
                else:
                    depth = max(cnot_swapping_gate_count(delta_c), 3*delta_t2)
            num_hops_array.append(num_hops)
            depth_array.append(depth)
            
        if min_metric == 'depth':
            min_depth = min(depth_array)
            indices_min_depth = [i for i, v in enumerate(depth_array) if v == min_depth]
            if len(indices_min_depth) > 1:
                num_hops_min_depth = list(np.array(num_hops_array)[indices_min_depth])
                min_hops_min_depth = np.argmin(num_hops_min_depth)
                m = indices_min_depth[min_hops_min_depth] + 1
            else:
                m = indices_min_depth[0] + 1
        else:
            min_hops = min(num_hops_array)
            indices_min_hops = [i for i, v in enumerate(num_hops_array) if v == min_hops]
            if len(indices_min_hops) > 1:
                depth_min_hops = list(np.array(depth_array)[indices_min_hops])
                min_depth_min_hops = np.argmin(depth_min_hops)
                m = indices_min_depth[min_depth_min_hops] + 1
            else:
                m = indices_min_hops[0] + 1
            
        return m
    
    if order == 'tct':
        m = n1+1
        for i in range(m-1):
            qc.cx(i,i+1)
            qc.cx(i+1,i)
            qc.cx(i,i+1)
        
        for i in range((n1+n2+2)-(m+1)):
            qc.cx(n1+n2+2-i,n1+n2+1-i)
            qc.cx(n1+n2+1-i,n1+n2+2-i)
            qc.cx(n1+n2+2-i,n1+n2+1-i)
            
        qc.cswap(m,m-1,m+1)
        
        for i in reversed(range((n1+n2+2)-(m+1))):
            qc.cx(n1+n2+2-i,n1+n2+1-i)
            qc.cx(n1+n2+1-i,n1+n2+2-i)
            qc.cx(n1+n2+2-i,n1+n2+1-i)
            
        for i in reversed(range(m-1)):
            qc.cx(i,i+1)
            qc.cx(i+1,i)
            qc.cx(i,i+1)
    else:
        if order == 'ttc':
            n1_save = n1
            n1 = n2
            n2 = n1_save
        
        if n1 != 0 or n2 != 0:
            m = optimal_m_Fredkin(n1, n2, min_metric)
        else:
            m = 1
        
        if n1 == 0:
            for i in range(m-1):
                qc.cx(2+i,1+i)
            for i in range(m-1):
                qc.cx(1+i,2+i)
                
            for i in range(m-1):
                qc.cx(1+i,i)
            for i in range(m-1):
                qc.cx(i,1+i)
        else:
            if m < n1 + 1:
                for i in range((n1 + 1) - m):
                    qc.cx(n1+1-i,n1-i)
                    qc.cx(n1-i,n1+1-i)
                    qc.cx(n1+1-i,n1-i)
            else:
                for i in range(m - (n1 + 1)):
                    qc.cx(n1+2+i,n1+1+i)
                    qc.cx(n1+1+i,n1+2+i)
                    qc.cx(n1+2+i,n1+1+i)
            
            for i in range(m-1):
                qc.cx(i,i+1)
                qc.cx(i+1,i)
                qc.cx(i,i+1)
    
        for i in range((n2+n1+2)-(m+1)):
            qc.cx((n2+n1+1)-i,(n2+n1+2)-i)
        for i in range((n2+n1+2)-(m+1)):
            qc.cx((n2+n1+2)-i,(n2+n1+1)-i)
            
        qc.cswap(m+1,m,m-1)
        
        for i in reversed(range((n2+n1+2)-(m+1))):
            qc.cx((n2+n1+2)-i,(n2+n1+1)-i)
        for i in reversed(range((n2+n1+2)-(m+1))):
            qc.cx((n2+n1+1)-i,(n2+n1+2)-i)
            
        if n1 == 0:
            for i in reversed(range(m-1)):
                qc.cx(i,1+i)
            for i in reversed(range(m-1)):
                qc.cx(1+i,i)
            
            for i in reversed(range(m-1)):
                qc.cx(1+i,2+i)
            for i in reversed(range(m-1)):
                qc.cx(2+i,1+i)
        else:
            for i in reversed(range(m-1)):
                qc.cx(i,i+1)
                qc.cx(i+1,i)
                qc.cx(i,i+1)
                
            if m < n1 + 1:
                for i in reversed(range((n1 + 1) - m)):
                    qc.cx(n1+1-i,n1-i)
                    qc.cx(n1-i,n1+1-i)
                    qc.cx(n1+1-i,n1-i)
            else:
                for i in reversed(range(m - (n1 + 1))):
                    qc.cx(n1+2+i,n1+1+i)
                    qc.cx(n1+1+i,n1+2+i)
                    qc.cx(n1+2+i,n1+1+i)
        
        if order == 'ttc':
            qc = qc.reverse_bits()
        
    return qc

def long_range_Toffoli_lin_con(n1, n2, order='cct', min_metric='depth'):
    """
    Function that outputs circuit that realizes Toffoli gate on trio of
    non-adjacent qubits in architecture restricted to linear connectivity.
    Specifically, target-qubit is in position 0, separated by n1 qubits
    from the nearest control-qubit (at position n1+1), which is separated
    by n2 qubits from the other control-qubit (at the most significant
    position n1+n2+2). This corresponds to the default order = 'cct'.
    If target-qubit is meant to be at the central position, order = 'ctc'.
    If target-qubit is meant to be at the most significant position, order
    = 'tcc'. The only difference between the three cases is a pair of
    Hadamard gates at the start and end of the circuit.
    If min_metric == 'depth', the circuit provided minimizes the depth,
    otherwise it minimizes the CNOT count. If there are multiple circuits
    with optimal depth (CNOT count), the one with lowest CNOT count (depth)
    is chosen, respectively.
    Follows P. M. Q. Cruz and B. Murta, APL Quantum 1, 016105 (2024). See, 
    in particular, Sec. III and Fig. 5.
    """
    
    # Subfunction that determines optimal intermediate positions,
    # i.e., after qubit rerouting, Toffoli gate is applied at trio
    # of adjacent qubits (m-1,m,m+1)
    def optimal_m_Toffoli(n1, n2, min_metric):
        
        def cnot_swapping_gate_count(s):
            if s == 1:
                return 2
            else:
                return 2+s
        
        num_hops_array = []
        depth_array = []
        for m in range(1,n2+n1+2):
            delta_c1 = n2 + n1 + 2 - (m + 1)
            delta_c2 = abs(n1 + 1 - m)
            delta_t = m - 1
            
            num_hops = 2*delta_c1 + 2*delta_c2 + 2*delta_t
            if m < n1 + 1: 
                if n2 == 0:
                    depth = max(cnot_swapping_gate_count(delta_c1) + 2, cnot_swapping_gate_count(delta_t))
                else:
                    depth = max(cnot_swapping_gate_count(delta_c1), cnot_swapping_gate_count(delta_t))
            else:
                if n1 == 0:
                    depth = max(cnot_swapping_gate_count(delta_c1), cnot_swapping_gate_count(delta_t) + 2)
                else:
                    depth = max(cnot_swapping_gate_count(delta_c1), cnot_swapping_gate_count(delta_t))
            num_hops_array.append(num_hops)
            depth_array.append(depth)
            
        if min_metric == 'depth':
            min_depth = min(depth_array)
            indices_min_depth = [i for i, v in enumerate(depth_array) if v == min_depth]
            if len(indices_min_depth) > 1:
                num_hops_min_depth = list(np.array(num_hops_array)[indices_min_depth])
                min_hops_min_depth = np.argmin(num_hops_min_depth)
                m = indices_min_depth[min_hops_min_depth] + 1
            else:
                m = indices_min_depth[0] + 1
        else:
            min_hops = min(num_hops_array)
            indices_min_hops = [i for i, v in enumerate(num_hops_array) if v == min_hops]
            if len(indices_min_hops) > 1:
                depth_min_hops = list(np.array(depth_array)[indices_min_hops])
                min_depth_min_hops = np.argmin(depth_min_hops)
                m = indices_min_depth[min_depth_min_hops] + 1
            else:
                m = indices_min_hops[0] + 1
            
        return m

    if n1 != 0 or n2 != 0:
        m = optimal_m_Toffoli(n1, n2, min_metric)
    else:
        m = 1

    qc = QuantumCircuit(n1+n2+3)
    
    if order == 'ctc':
        qc.h(0)
        qc.h(n1+1)
    elif order == 'tcc':
        qc.h(0)
        qc.h(n1+n2+2)
        
    if n1+1 > m:
        for i in range(n1+1-m):
            qc.cx(n1-i, n1+1-i)
        for i in range(n1+1-m):
            qc.cx(n1+1-i, n1-i)
    elif n1+1 < m:
        for i in range(m-n1-1):
            qc.cx(n1+2+i, n1+1+i)
        for i in range(m-n1-1):
            qc.cx(n1+1+i, n1+2+i)
    
    for i in range(m-1):
        qc.cx(i,i+1)
    for i in range(m-1):
        qc.cx(i+1,i)
    
    for i in range(n2+n1+1-m):
        qc.cx(n1+n2+1-i,n1+n2+2-i)
    for i in range(n2+n1+1-m):
        qc.cx(n1+n2+2-i,n1+n2+1-i)
    
    qc.ccx(m+1,m,m-1)
    
    for i in reversed(range(n2+n1+1-m)):
        qc.cx(n1+n2+2-i,n1+n2+1-i)
    for i in reversed(range(n2+n1+1-m)):
        qc.cx(n1+n2+1-i,n1+n2+2-i)
    
    for i in reversed(range(m-1)):
        qc.cx(i+1,i)
    for i in reversed(range(m-1)):
        qc.cx(i,i+1)
        
    if n1+1 > m:
        for i in reversed(range(n1+1-m)):
            qc.cx(n1+1-i, n1-i)
        for i in reversed(range(n1+1-m)):
            qc.cx(n1-i, n1+1-i)
    elif n1+1 < m:
        for i in reversed(range(m-n1-1)):
            qc.cx(n1+1+i, n1+2+i)
        for i in reversed(range(m-n1-1)):
            qc.cx(n1+2+i, n1+1+i)
            
    if order == 'ctc':
        qc.h(0)
        qc.h(n1+1)
    elif order == 'tcc':
        qc.h(0)
        qc.h(n1+n2+2)
        
    return qc

def multiplexor_basis_gate_decomp(gate_label, params, num_qubits, 
                                  starting_call=True, cancel_cnots=False, 
                                  forward=True):
    """
    Basis gate decomposition of direct sum of Rz or Ry single-qubit gates
    based on Theorems 4 and 8 from "Synthesis of Quantum Logic Circuits",
    V. V. Shende, S. S. Bullock and I. L. Markov, arXiv:quant-ph/0406176v5.
    Gate label is 'y' or 'z', depending on type of rotation performed.
    angles is the array of parameter of the rotation gates. num_qubits is 
    the total number of qubits (i.e., length of angles is 2**(num_qubits-1)) 
    on which the multiplexor acts. Outputs circuit in Qiskit format.
    """
    
    if starting_call:
        transform = Walsh_Hadamard_trans_matrix(num_qubits-1, no_prefactor=True)
        params = 1/2**(num_qubits-1)*transform.dot(params)
    
    qc = QuantumCircuit(num_qubits)
    
    if num_qubits == 2:
        if forward:
            exec('qc.r'+gate_label+'(params[0], 0)')
            qc.cx(1,0)
            exec('qc.r'+gate_label+'(params[1], 0)')
            if not cancel_cnots:
                qc.cx(1,0)
        else:
            if not cancel_cnots:
                qc.cx(1,0)
            exec('qc.r'+gate_label+'(params[1], 0)')
            qc.cx(1,0)
            exec('qc.r'+gate_label+'(params[0], 0)')
    else:
        list_of_qubits = list(range(num_qubits-1))
        if forward:
            subcirc1 = multiplexor_basis_gate_decomp(gate_label, params[:int(2**(num_qubits-2))], num_qubits-1, False, True, True)
            qc.append(subcirc1, list_of_qubits)
            qc.cx(num_qubits-1, 0)
            subcirc2 = multiplexor_basis_gate_decomp(gate_label, params[int(2**(num_qubits-2)):], num_qubits-1, False, True, False)
            qc.append(subcirc2, list_of_qubits)
            if starting_call:
                qc.cx(num_qubits-1, 0)
        else:
            if starting_call:
                qc.cx(num_qubits-1, 0)
            subcirc1 = multiplexor_basis_gate_decomp(gate_label, params[int(2**(num_qubits-2)):], num_qubits-1, False, True, True)
            qc.append(subcirc1, list_of_qubits)
            qc.cx(num_qubits-1, 0)
            subcirc2 = multiplexor_basis_gate_decomp(gate_label, params[:int(2**(num_qubits-2))], num_qubits-1, False, True, False)
            qc.append(subcirc2, list_of_qubits)
   
    return qc

def permutation_qcircuit(permutation_list, connectivity='all'):
    """
    Function that generates quantum circuit that permutes qubits
    according to permutation_list provided as input. For example,
    [2,0,1] is a cyclic permutation where Q0 -> Q2, Q1 -> Q0 and 
    Q2 -> Q1. The length of permutation_list determines the number 
    of qubits of the circuit, so all qubits must be explicitly 
    included in the list even if their position remains unchanged. 
    For example, [0,1,2,3,5,4] would be a permutation where the
    only nontrivial action is Q4 <-> Q5. 
    If connectivity == 'all', circuit has depth at most 2 SWAPs, 
    or 6 CNOTs, following the method introduced in Sec. 4.3.2 of
    Bruno Murta's Ph.D. dissertation. If connectivity = 'lin', 
    then only SWAPs between adjacent qubits are considered via 
    the Amida lottery method introduced by Seki, Shirakawa and
    Yunoki in Sec. III.C of Phys. Rev. A 101, 052340 (2020).
    Circuit is decomposed in terms of SWAPs.
    """
    
    # subfunction that finds disjoint cycles of permutation
    def disjoint_cycles_generator(permutation_list):
        disjoint_cycles_list = []
        elements_already_in_disjoint_cycles = []
         
        for element_i in range(len(permutation_list)):
            if element_i not in elements_already_in_disjoint_cycles:
                new_disjoint_cycle = [element_i]
                elements_already_in_disjoint_cycles.append(element_i)
                new_position = permutation_list[element_i]
                while new_position != element_i:
                    new_disjoint_cycle.append(new_position)
                    elements_already_in_disjoint_cycles.append(new_position)
                    new_position = permutation_list[new_position]
                disjoint_cycles_list.append(new_disjoint_cycle)
        
        return disjoint_cycles_list
    
    # subfunction that finds network of SWAP gates between
    # adjacent qubits via Amida lottery method
    def Amida_lottery_method(permutation_list):
        initial_positions = []
        for i in range(len(permutation_list)):
            initial_positions.append((i,0))
        final_positions = []
        for i in permutation_list:
            final_positions.append((i,1))
        list_of_intersections_y_values = []
        list_of_intersections_labels = []
        
        for i in range(len(permutation_list)):
            for j in range(i, len(permutation_list)):
                line_1 = (initial_positions[i], final_positions[i])
                line_2 = (initial_positions[j], final_positions[j])
                intersection = line_intersection(line_1, line_2, [0,1])
                if intersection:
                    list_of_intersections_y_values.append(intersection[1])
                    list_of_intersections_labels.append([i,j])
        
        list_of_SWAP_labels = sorting_list_according_to_order_of_another_list(list_of_intersections_labels, list_of_intersections_y_values)
        
        for i in range(len(list_of_SWAP_labels)):
            current_SWAP = list_of_SWAP_labels[i]
            for j in range(i+1,len(list_of_SWAP_labels)):
                for k in range(2):
                    if list_of_SWAP_labels[j][k] == current_SWAP[0]:
                        list_of_SWAP_labels[j][k] = current_SWAP[1]
                    elif list_of_SWAP_labels[j][k] == current_SWAP[1]:
                        list_of_SWAP_labels[j][k] = current_SWAP[0]
        
        return list_of_SWAP_labels
    
    num_qubits = len(permutation_list)
    qc = QuantumCircuit(num_qubits)
    
    if connectivity == 'all':
        disjoint_cycles_list = disjoint_cycles_generator(permutation_list)
        for disjoint_cycle in disjoint_cycles_list:
            for i in range(int(np.floor(len(disjoint_cycle)/2))):
                qc.swap(disjoint_cycle[i],disjoint_cycle[len(disjoint_cycle)-1-i])
            disjoint_cycle = disjoint_cycle[1:]
            for i in range(int(np.floor(len(disjoint_cycle)/2))):
                qc.swap(disjoint_cycle[i],disjoint_cycle[len(disjoint_cycle)-1-i])
    else:
        list_of_SWAP_labels = Amida_lottery_method(permutation_list)
        for i in range(len(list_of_SWAP_labels)):
            Q0 = list_of_SWAP_labels[i][0]
            Q1 = list_of_SWAP_labels[i][1]
            qc.swap(Q0,Q1)
    
    return qc