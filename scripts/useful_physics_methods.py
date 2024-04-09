import numpy as np, scipy as sp, math as mt, cmath as cmt, qiskit, random, itertools
from qiskit import *
from quspin.operators import hamiltonian
from quspin.basis import tensor_basis, spin_basis_1d

def AKLT_Hamiltonian(N_site, N, coupling_map):
    '''
    Matrix representation of AKLT Hamiltonian for a given N-site lattice with coordination
    number N_site, such that the local degree of freedom at each site is a spin-N_site/2.
    The coupling_map input determines the pairs of spins that have an AKLT term between them.
    '''
    
    if N_site % 2 == 0:
        basis = spin_basis_1d(N,pauli=False,S = str(int(N_site/2)))
    else:
        basis = spin_basis_1d(N,pauli=False,S = str(N_site)+'/2')
    
    # Coefficients of expansion of NN coupling in terms of powers of BL term
    coefficients = coefficients_BL_powers_AKLT_model(N_site)
    
    # define operators according to coupling_map
    J_zz = []
    J_xy = []
    for i in range(len(coupling_map)):
        new_coupling = coupling_map[i]
        J_zz.append([1,new_coupling[0],new_coupling[1]])
        J_xy.append([1/2,new_coupling[0],new_coupling[1]])
        
    # First NN pair to initialize sparse representation of Hamiltonian
    static = [["+-",[J_xy[0]]],["-+",[J_xy[0]]],["zz",[J_zz[0]]]]
    dynamic=[]
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    bilinear_term = hamiltonian(static,dynamic,basis=basis,dtype=np.float64, **no_checks).static
    full_Hamiltonian = coefficients[0] * bilinear_term
    current_BL_power = bilinear_term
    for i in range(1, len(coefficients)):
        current_BL_power = current_BL_power @ bilinear_term
        full_Hamiltonian = full_Hamiltonian + coefficients[i] * current_BL_power
        
    # All remaining NN pairs
    for i in range(1,len(J_zz)):
        static = [["+-",[J_xy[i]]],["-+",[J_xy[i]]],["zz",[J_zz[i]]]]
        dynamic=[]
        no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
        bilinear_term = hamiltonian(static,dynamic,basis=basis,dtype=np.float64, **no_checks).static
        full_Hamiltonian = full_Hamiltonian + coefficients[0] * bilinear_term
        current_BL_power = bilinear_term
        for i in range(1, len(coefficients)):
            current_BL_power = current_BL_power @ bilinear_term
            full_Hamiltonian = full_Hamiltonian + coefficients[i] * current_BL_power

    return np.real(full_Hamiltonian)

def coefficients_BL_powers_AKLT_model(N_site):
    '''
    Coefficients of expansion of single AKLT term in terms of powers of
    bilinear term for a given local spin-N_site/2 operator. By convention,
    the additive constant is removed and the prefactor of the bilinear term
    is set to unity. Coefficients are ordered in increasing powers of BL term.
    '''
    
    # Generating auxiliary terms
    aux_terms = []
    for i in range(N_site):
        new_aux_term = [N_site*(N_site/2+1) - i*(i+1), 2]
        aux_terms.append(new_aux_term)
            
    # Adding auxiliary terms to obtain coefficients prior to rescaling and reshifting
    coefficients = []
    terms_indices_list = list(range(N_site))
    for i in range(N_site+1):
        terms_subsets = list(itertools.combinations(terms_indices_list, i))
        new_coefficient = 0
        for j in range(len(terms_subsets)):
            one_term_of_coefficient = 1
            for k in range(N_site):
                if k in terms_subsets[j]:
                    one_term_of_coefficient = one_term_of_coefficient * aux_terms[k][1]
                else:
                    one_term_of_coefficient = one_term_of_coefficient * aux_terms[k][0]
            new_coefficient = new_coefficient + one_term_of_coefficient
        coefficients.append(new_coefficient)
    
    # Removing constant term and setting prefactor of BL term to unity 
    coefficients = [x/coefficients[1] for x in coefficients[1:]]
    
    return coefficients

def converting_auxiliaries_into_physical_at_all_sites(input_state_as_tensor, 
                                                      sets_of_qubits_encoding_a_site, 
                                                      N_site):
    '''
    Function that takes as input a tensor with as many 2-dim indices as the 
    number of qubits of the lattice and outputs the vectorized form of a tensor
    with as many (N_site+1)-dim indices as the number of sites of the lattice, 
    where each lattice site is encoded in terms of N_site qubits.
    
    Inputs:
    input_state_as_tensor: Rank-N tensor with all indices having dimension 2, 
                           where N is the total number of qubits
    sets_of_qubits_encoding_a_site: list with num_sites entries (where num_sites 
                                    is the number of lattice sites), each
                                    corresponding to the indices of the N_site 
                                    qubits that encode a site. The order of
                                    the entries of this list is assumed to be 
                                    the order of the sites in the output.
    N_site: number of qubits encoding a single lattice site.
    
    Outputs:
    output_state_as_tensor: Same state encoded by input_state_as_tensor but with 
                            num_sites indices of dimension N_site+1 instead of N 
                            indices of dimension 2. 
    output_state_as_vector: Same as output_state_as_tensor, but reshaped into 
                            vector of dimension (N_site+1)^num_sites. 
    '''
    
    # Obtaining matrix representation of sitewise converter
    sitewise_converter = N_spin_1_2_basis_to_spin_N_2_basis_conversion_matrix(N_site)
    
    # Defining order of qubits
    num_sites = len(sets_of_qubits_encoding_a_site) # Total number of lattice sites
    order_of_qubits = []
    for i in range(num_sites):
        for j in range(N_site):
            order_of_qubits.append(sets_of_qubits_encoding_a_site[i][j])
    N = input_state_as_tensor.ndim  # Total number of qubits 
            
    # Reordering qubits such that all qubits associated with the same site are 
    # next to one another As noted above, the order of the sites is determined 
    # by the order followed in sets_of_qubits_encoding_a_site
    output_state_as_tensor = np.transpose(input_state_as_tensor, order_of_qubits)
    
    # Application of sitewise converter one site at a time to avoid storing large matrix
    for i in range(num_sites):
        
        # Applying sitewise converter to current state
        converter_dot_output_state = np.outer(sitewise_converter, output_state_as_tensor)
        dim_last_index = 2**(N-N_site*(i+1)) * (N_site+1)**i
        converter_dot_output_state = np.reshape(converter_dot_output_state, (N_site+1, 2**N_site, 2**N_site, dim_last_index))
        output_state_as_tensor = np.einsum(converter_dot_output_state, [0,1,1,2], [0,2])
        
        # Moving the current physical index to the end
        output_state_as_tensor = np.reshape(output_state_as_tensor, (N_site+1,) + (2,)*(N-N_site*(i+1)) + (N_site+1,)*i)
        move_first_index_to_end = list(range(output_state_as_tensor.ndim))
        move_first_index_to_end.append(move_first_index_to_end.pop(move_first_index_to_end.index(0)))
        output_state_as_tensor = np.transpose(output_state_as_tensor, move_first_index_to_end)
        
    # Obtaining vectorized form of output
    output_state_as_vector = np.reshape(output_state_as_tensor, ((N_site+1)**num_sites,))
    
    return output_state_as_vector, output_state_as_tensor

def is_left_or_right_normalized(A, which):
    """
    Function that determines if a given rank-3 tensor A is left-normalized,
    right-normalized or both (i.e., unitary). The respective 'which' inputs
    are 'left', 'right', and 'both'.
    """
    
    [Dleft, d, Dright] = A.shape
    
    if which == 'left' or which == 'both':
        A = np.reshape(A, (Dleft*d, Dright))
        Adag = np.transpose(np.conjugate(A))
        if np.allclose(Adag.dot(A), np.eye(Dright)):
            outcome_left = True
        else:
            outcome_left = False
        
    if which == 'right' or which == 'both':
        A = np.reshape(A, (Dleft, Dright*d))
        Adag = np.transpose(np.conjugate(A))
        if np.allclose(A.dot(Adag), np.eye(Dleft)):
            outcome_right = True
        else:
            outcome_right = False
            
    if which == 'left':
        return outcome_left
    elif which == 'right':
        return outcome_right
    else:
        return outcome_left and outcome_right

def left_canonical_MPS(MPS_arr, return_norm=False):
    """
    Function that turns given MPS with open boundary conditions
    into its left-canonical form. Outputs only left-canonical
    MPS if return_norm == False, otherwise it also returns the
    norm prior to the normalization.
    """
    
    N = len(MPS_arr)
    
    for i in range(N):
        current_tensor_saved = MPS_arr[i]
        (Dleft,dsite,Dright) = current_tensor_saved.shape
        current_tensor_reshaped = np.reshape(current_tensor_saved,(Dleft*dsite,Dright))
        U, S, Vdagger = np.linalg.svd(current_tensor_reshaped, full_matrices=False)
        MPS_arr[i] = U.reshape(((Dleft,dsite,int(U.size/(Dleft*dsite)))))
        if i < N-1:
            leftover = np.matmul(np.diag(S),Vdagger)
            MPS_arr[i+1] = np.tensordot(leftover,MPS_arr[i+1],axes=([1],[0]))
    
    if return_norm:
        return MPS_arr, S[0]
    else:
        return MPS_arr

def MPS_canonical_examples(key, n):
    """
    Function that outputs matrix product state (MPS) representation
    of canonical examples of states that can be exactly represented
    as an MPS with small bond dimension. Input 'n' is number of local
    degrees of freedom. Input 'key' can take the following values: 
    
    'GHZ'           : Greenberger-Horne-Zeilinger state as MPS with PBCs.
    'MG'            : Majumdar-Ghosh model ground state as MPS with PBCs.
    'VBS-PBC'       : Spin-1 AKLT model ground state as MPS with PBCs. 
    'VBS-OBC'       : Spin-1 AKLT model ground state as MPS with OBCs.
    'VBS-PBC-qubits': Spin-1 AKLT model ground state as MPS with PBCs and two qubits per site. 
    'VBS-OBC-qubits': Spin-1 AKLT model ground state as MPS with OBCs and two qubits per site.
    'W'             : W state (Dicke state with Hamming weight 1) as MPS with PBCs. 
    
    All states are normalized.
    """
    
    if key == 'GHZ':
        A = np.zeros((2,2,2))
        A[:,0,:] = 1/(np.sqrt(2))**(1/n) * np.array([[1,0],
                                                     [0,0]])
        A[:,1,:] = 1/(np.sqrt(2))**(1/n) * np.array([[0,0],
                                                     [0,1]])
        MPS_arr = [A]*n
        
    if key == 'MG':
        if n % 2 != 0:
            print('n must be even!')
            return -1
        else:
            VB = 1/np.sqrt(2)*np.array([[ 0,1],
                                        [-1,0]])
            U, S, Vdag = np.linalg.svd(VB)
            A1 = U.reshape((1,2,2))
            A2 = np.diag(S).dot(Vdag).reshape((2,2,1))
            MPS_arr = [A1, A2]*(int(n/2))
        
    if key == 'VBS-OBC' or key == 'VBS-PBC':
        A = np.zeros((2,3,2))
        A[:,0,:] = np.array([[0, np.sqrt(2/3)],
                             [0,            0]])
        A[:,1,:] = np.array([[-1/np.sqrt(3), 0],
                             [0,  1/np.sqrt(3)]])
        A[:,2,:] = np.array([[0,             0],
                             [-np.sqrt(2/3), 0]])

        if key == 'VBS-PBC':
            MPS_arr = [A]*n
        else:
            MPS_arr = [2**(1/4)*A[0,:,:].reshape((1,3,2))] + [A]*(n-2) + [2**(1/4)*A[:,:,0].reshape((2,3,1))]
            
    if key == 'VBS-OBC-qubits' or key == 'VBS-PBC-qubits':
        A = np.zeros((2,4,2))
        A[:,0,:] = np.array([[0, np.sqrt(2/3)],
                             [0,            0]])
        A[:,1,:] = np.array([[-1/np.sqrt(6), 0],
                             [0,  1/np.sqrt(6)]])
        A[:,2,:] = np.array([[-1/np.sqrt(6), 0],
                             [0,  1/np.sqrt(6)]])
        A[:,3,:] = np.array([[0,             0],
                             [-np.sqrt(2/3), 0]])

        if key == 'VBS-PBC-qubits':
            MPS_arr = [A]*n
        else:
            MPS_arr = [2**(1/4)*A[0,:,:].reshape((1,4,2))] + [A]*(n-2) + [2**(1/4)*A[:,:,0].reshape((2,4,1))]
    
    if key == 'W':
        A_left = np.zeros((2,2,2))
        A_left[:,0,:] = 1/(np.sqrt(n))**(1/n) * np.array([[1,0],
                                                          [1,0]])
        A_left[:,1,:] = 1/(np.sqrt(n))**(1/n) * np.array([[0,0],
                                                          [0,1]])
        
        A_bulk = np.zeros((2,2,2))
        A_bulk[:,0,:] = 1/(np.sqrt(n))**(1/n) * np.array([[1,0],
                                                          [0,1]])
        A_bulk[:,1,:] = 1/(np.sqrt(n))**(1/n) * np.array([[0,1],
                                                          [0,0]])
        
        A_right = np.zeros((2,2,2))
        A_right[:,0,:] = 1/(np.sqrt(n))**(1/n) * np.array([[0,0],
                                                           [0,1]])
        A_right[:,1,:] = 1/(np.sqrt(n))**(1/n) * np.array([[1,0],
                                                           [0,0]])
        
        MPS_arr = [A_left] + [A_bulk]*(n-2) + [A_right]
        
    return MPS_arr

def MPS_full_contraction(MPS_arr):
    """
    Function that contracts given matrix product state
    (MPS), thus generating the corresponding statevector.
    Valid for both open and periodic boundary conditions.
    """
    
    N = len(MPS_arr)
    
    psi = MPS_arr[0]
    for i in range(1,N):
        psi = np.tensordot(psi,MPS_arr[i],axes=([1+i],[0]))
    axis_for_trace = list(range(N+1))+[0]
    psi = np.einsum(psi, axis_for_trace, list(range(1,N+1)))
    psi = np.reshape(psi, (psi.size,))
    
    return psi

def MPS_OBC_generator_from_state(psi, n, d, D):
    """
    Function that generates matrix product state (MPS)
    with n sites, local Hilbert space dimension d, and
    bond dimension cutoff D given statevector psi, which
    must be (d^n)-dimensional array.
    
    Resulting MPS is left-normalized, as it is constructed
    following a left-to-right sweep.
    """
    
    psi = np.reshape(psi, (d,)*n)

    M = psi
    MPS_array = []
    dim1 = 1
    for i in range(n-1):
        M = np.reshape(M, (int(dim1*d), int(d**(n-1-i))))
        U, S, Vdag = np.linalg.svd(M, full_matrices=False)

        S = S[:D]
        U = U[:,:D]
        Vdag = Vdag[:D,:]

        A = np.reshape(U, (int(dim1), d, int(U.size/(dim1*d))))
        MPS_array.append(A)

        dim1 = U.size/(d*dim1)
        M = np.reshape(np.diag(S).dot(Vdag), (int(dim1), int(d**(n-1-i))))

    M = np.reshape(M, (M.shape[0], M.shape[1], 1))
    MPS_array.append(M)
    
    return MPS_array

def norm_MPS(MPS_arr, BC):
    '''
    Function that determines norm of state encoded in matrix product state
    provided through input MPS_arr. BC stands for "boundary conditions" and
    can take values 'OBC' (open) or PBC (periodic).
    '''
    
    if BC == 'PBC':
        N = len(MPS_arr)
        for i in range(N):
            [Dleft, d, Dright] = MPS_arr[i].shape
            T_i = np.tensordot(MPS_arr[i], np.conjugate(MPS_arr[i]), axes=([1],[1]))
            T_i = np.transpose(T_i, (0,2,1,3))
            T_i = np.reshape(T_i, (Dleft*Dleft, Dright*Dright))
            if i == 0:
                T_product = T_i
            else:
                T_product = T_product.dot(T_i)
    
        norm = np.sqrt(np.trace(T_product))
    else:
        lc_MPS_arr, norm = left_canonical_MPS(MPS_arr, return_norm=True)
    
    return norm

def N_spin_1_2_basis_to_spin_N_2_basis_conversion_matrix(N):
    '''
    Function that generates (N+1) x 2^N matrix that converts the computational basis 
    of N spins-1/2 into the computational basis of a single spin-N/2
    '''
    
    # Generating the 2^N x 2^N matrix representation of the lowering operator S^{-}
    # for the sum of N spins-1/2
    Sx = 0.5*np.array([[0, 1],
                       [1, 0]])
    Sy = 0.5*np.array([[0, -1j],
                       [1j, 0]])

    SxTotal = np.zeros((2**N, 2**N), dtype=complex)
    SyTotal = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        SxTotal = SxTotal + np.kron(np.eye(2**(N-i-1)), np.kron(Sx, np.eye(2**i)))
        SyTotal = SyTotal + np.kron(np.eye(2**(N-i-1)), np.kron(Sy, np.eye(2**i)))
        
    SminusTotal = SxTotal - 1j * SyTotal
    
    # Applying the lowering operator S^{-} to |N/2, N/2) to generate all other 
    # computational basis states for a spin-N/2 in the original basis spanned 
    # by the N spins-1/2
    eig_N_2_N_2 = np.zeros((2**N,), dtype=complex)
    eig_N_2_N_2[0] = 1
    
    U = np.zeros((N+1, 2**N), dtype=complex)
    U[0,:] = eig_N_2_N_2
    
    current_eig = eig_N_2_N_2
    for i in range(N):
        current_eig = SminusTotal.dot(current_eig)
        current_eig = 1/np.linalg.norm(current_eig)*current_eig
        U[i+1,:] = current_eig

    return U

def random_MPS_generator_OBC(n, d, D):
    """
    Function that outputs a list with the n local tensors of
    a random matrix product state (MPS) with n free indices of
    dimension d and bond dimension D. Boundary conditions BC
    are either open ('OBC') or periodic ('PBC').
    """
    
    MPS_array = []
    
    M = np.random.rand(1,d,D)
    MPS_array.append(M)
    
    for i in range(1,n-1):
        M = np.random.rand(D,d,D)
        MPS_array.append(M)
        
    M = np.random.rand(D,d,1)
    MPS_array.append(M)
    
    return MPS_array

def random_MPS_generator_PBC(N, d, D, which, trans_invariant=True):
    '''
    Function that generates random matrix product state (MPS) with periodic boundary
    conditions. MPS is normalized, so the local tensors are left-normalized (if 
    which == 'left') or right-normalized (if which == 'right'). 
    If trans_invariant == True, all local tensors are equal, otherwise each is 
    generated separately.
    '''
    
    if trans_invariant:
        new_unitary = sp.stats.ortho_group.rvs(D*d)
        if which == 'left':
            new_tensor = new_unitary[:,:D].reshape((D,d,D))
        elif which == 'right':
            new_tensor = new_unitary[:D,:].reshape((D,d,D))
        MPS_arr = [new_tensor]*N
    else:
        MPS_arr = []
        for i in range(N):
            new_unitary = sp.stats.ortho_group.rvs(D*d)
            if which == 'left':
                new_tensor = new_unitary[:,:D].reshape((D,d,D))
            elif which == 'right':
                new_tensor = new_unitary[:D,:].reshape((D,d,D))
            MPS_arr.append(new_tensor)
        
    return MPS_arr 

def symmetrization(N):
    '''
    Function that returns 2^N x 2^N matrix representation of symmetrization operator
    that converts N spins-1/2 into a single spin-N/2 but still represented in basis
    spanned by N spins-1/2
    '''
    
    def split(word): 
        return [char for char in word]
    
    Symm_rep = np.zeros((2**N, 2**N))
    for i in range(2**N):
        bin_i = bin(i)[2:]
        while len(bin_i) < N:
            bin_i = '0' + bin_i
        bin_i_split = split(bin_i)
        perms = list(itertools.permutations(bin_i_split))
        for j in range(len(perms)):
            aux = ''
            for k in range(len(perms[j])):
                aux = aux + perms[j][k]
            aux = int(aux, 2)
            Symm_rep[i, aux] += 1/mt.factorial(N)
    return Symm_rep





