import numpy as np, scipy as sp, math as mt, cmath as cmt, qiskit, random
from qiskit import *
from qiskit.circuit.library import RYGate, RZGate
from useful_computing_methods import *
from useful_maths_methods import *
from useful_physics_methods import *
from unitary_decompositions import *

def MPS_preparation_quantum_circuit_OBC(MPS_arr):
    """
    Function that generates quantum circuit to prepare a
    given MPS with open boundary conditions. Circuit consists 
    of application of inverse of matrix product disentanglers 
    (MPDs) in reverse order.
    
    MPS is assumed to be provided as a list of numpy.arrays that
    are rank-3 tensors with the virtual index corresponding to
    the bond to the left as the first index, the physical index
    as the second index, and the virtual index corresponding to
    the bond to the right as the third index. The first and last
    local tensors should also be rank-3 through the addition of
    a dummy index of dimension 1.
    
    MPS does not have to be in left-canonical form. This is taken
    care of in the MPD_generator_OBC function.
    """
    
    def MPD_generator_OBC(MPS_arr):
        """
        Function that generates matrix product disentanglers (MPDs)
        for a given MPS with open boundary conditions. As the name 
        suggests, a MPD disentangles one site from the remainder of 
        the MPS. Hence, having prepared the MPS, applying all MPDs 
        in sequence generates the fiducial state |000...00).
        
        MPS is assumed to be provided as a list of numpy.arrays that
        are rank-3 tensors with the virtual index corresponding to
        the bond to the left as the first index, the physical index
        as the second index, and the virtual index corresponding to
        the bond to the right as the third index. The first and last
        local tensors should also be rank-3 through the addition of
        a dummy index of dimension 1.
        
        MPS does not have to be in left-canonical form.
        """
        
        MPS_lc = left_canonical_MPS(MPS_arr)
        MPD_array = []
        
        for i in range(len(MPS_lc)):
            (Dleft, d, Dright) = MPS_lc[i].shape
            A_i = np.reshape(MPS_lc[i], (Dleft*d,Dright))
            
            aux_dim_array = np.array([2**np.ceil(np.log2(Dleft*d)), 2**np.ceil(np.log2(Dright))])
            enlarged_dim = int(np.amax(aux_dim_array))
            A_i_enlarged = np.zeros((enlarged_dim,enlarged_dim), dtype=complex)
            A_i_enlarged[:(Dleft*d),:Dright] = A_i

            kernel_i = sp.linalg.null_space(np.transpose(np.conjugate(A_i_enlarged[:,:Dright])))
            A_i_enlarged[:,Dright:] = kernel_i
            
            MPD_array.append(np.transpose(np.conjugate(A_i_enlarged)))
            
        return MPD_array

    
    n = len(MPS_arr)                             # number of sites
    d = MPS_arr[0].shape[1]                      # local Hilbert space dimension
    qubits_per_site = int(np.ceil(np.log2(d)))   # number of qubits encoding local Hilbert space
    MPD_arr = MPD_generator_OBC(MPS_arr)         # array of matrix product disentanglers
    
    q = qiskit.QuantumRegister(n*qubits_per_site)
    qc = qiskit.QuantumCircuit(q)

    for i in range(n):
        U_i = np.transpose(np.conjugate(MPD_arr[n-1-i]))
        num_qubits_i = int(np.log2(U_i.shape[0]))
        MPDi_gate = qiskit.quantum_info.operators.Operator(U_i)
        qc.unitary(MPDi_gate, q[i*qubits_per_site:i*qubits_per_site+num_qubits_i])

    return qc

def MPS_preparation_quantum_circuit_PBC(MPS_arr):
    """
    Function that generates quantum circuit to prepare a
    given MPS with periodic boundary conditions. Circuit consists 
    of application of inverse of matrix product disentanglers 
    (MPDs) in reverse order. Desired state corresponding to MPS
    is successfully prepared if most significant qubit is measured
    in |0) in the computational basis, so the method is probabilistic.
    This projective measurement of the most significant qubit is not
    included in structure of circuit, as further coherent manipulations
    may follow this state preparation.
    
    MPS is assumed to be provided as a list of numpy.arrays that
    are rank-3 tensors with the virtual index corresponding to
    the bond to the left as the first index, the physical index
    as the second index, and the virtual index corresponding to
    the bond to the right as the third index.
    
    Before the MPDs are generated, SVD is carried out at all sites 
    except the first one to ensure the respective local tensors are 
    all left-normalized. The first one does not have to be left-normalized.
    
    The dimension of the physical indices is assumed to be a power of
    2 in order to be encoded naturally in terms of qubits, since the
    purpose of this function is to generate a quantum circuit. The
    bond dimensions can take any positive integral value, and so can
    the number of sites of the MPS.
    """
    
    def MPD_generator_PBC(MPS_arr):
        """
        Function that generates the matrix product disentanglers (MPDs)
        to disentangle one local degree of freedom at a time from the
        remainder of the MPS with periodic boundary conditions.
        """
        
        N = len(MPS_arr)
        
        MPD_arr = []

        [Dleft, d, Dright] = MPS_arr[0].shape
        enlarged_d = int(2**(mt.ceil(np.log2(d))))
        enlarged_Dleft = int(2**(mt.ceil(np.log2(Dleft))))
        enlarged_Dright = int(2**(mt.ceil(np.log2(Dright))))
        max_dim = max(enlarged_d, enlarged_Dleft * enlarged_Dright)
        num_qubits = mt.ceil(np.log2(max_dim))
        tensor_with_zeros = np.zeros((enlarged_Dleft, enlarged_d, enlarged_Dright), dtype=complex)
        tensor_with_zeros[:Dleft,:d,:Dright] = MPS_arr[0]
        inter = np.transpose(tensor_with_zeros, (1, 0, 2))
        inter = np.reshape(inter, (enlarged_d,enlarged_Dleft*enlarged_Dright))
        tlq = np.zeros((int(2**num_qubits),int(2**num_qubits)), dtype=complex)
        tlq[:enlarged_d,:(enlarged_Dleft*enlarged_Dright)] = inter

        u, s, vh = np.linalg.svd(tlq, full_matrices=True)
        smax = np.amax(s)
        if smax > 1:
            tlq = 1/(smax+0.001)*tlq
        
        A_1_enl = np.zeros((int(2**(num_qubits+1)),int(2**(num_qubits+1))), dtype=complex)
        A_1_enl[:int(2**num_qubits),:int(2**num_qubits)] = tlq

        uf, sf, vhf = np.linalg.svd(tlq, full_matrices=True)
        s_primedf = np.sqrt(np.ones((int(2**num_qubits))) - np.square(sf))
        blq = np.matmul(uf,np.matmul(np.diag(s_primedf),vhf))
        A_1_enl[int(2**num_qubits):,:int(2**num_qubits)] = blq

        k_1 = sp.linalg.null_space(np.conjugate(np.transpose(A_1_enl)))
        A_1_enl[:,int(2**num_qubits):] = k_1

        MPD1 = np.transpose(np.conjugate(A_1_enl))
        MPD_arr.append(MPD1)
        
        for site in range(1,N-1):
            [Dleft, d, Dright] = MPS_arr[site].shape
            A_site = np.reshape(MPS_arr[site], (Dleft*d, Dright))
            max_dim = max(Dleft*d, Dright)
            num_qubits = mt.ceil(np.log2(max_dim))
            A_site_enl = np.zeros((int(2**num_qubits),int(2**num_qubits)), dtype=complex)
            A_site_enl[:(Dleft*d),:Dright] = A_site 

            k_site = sp.linalg.null_space(np.conjugate(np.transpose(A_site_enl)))
            A_site_enl[:,Dright:] = k_site

            MPDsite = np.transpose(np.conjugate(A_site_enl))
            MPD_arr.append(MPDsite)
        
        num_qubits_left = mt.ceil(np.log2(MPS_arr[N-1].shape[0]))
        num_qubits_center = mt.ceil(np.log2(MPS_arr[N-1].shape[1]))
        num_qubits_right = mt.ceil(np.log2(MPS_arr[N-1].shape[2]))
        num_qubits = num_qubits_left + num_qubits_center + num_qubits_right    
        tensor_with_zeros = np.zeros((2**num_qubits_left, 2**num_qubits_center, 2**num_qubits_right), dtype=complex)
        tensor_with_zeros[:MPS_arr[N-1].shape[0],:MPS_arr[N-1].shape[1],:MPS_arr[N-1].shape[2]] = MPS_arr[N-1]
        vector_form = tensor_with_zeros.reshape((tensor_with_zeros.size,))
        vector_form = 1/np.linalg.norm(vector_form)*vector_form
        A_N_enl = np.zeros((int(2**num_qubits),int(2**num_qubits)), dtype=complex)
        A_N_enl[:tensor_with_zeros.size,0] = vector_form

        k_N = sp.linalg.null_space(np.conjugate(np.transpose(A_N_enl)))
        A_N_enl[:,1:] = k_N

        MPDN = np.transpose(np.conjugate(A_N_enl))
        MPD_arr.append(MPDN)
        
        return MPD_arr
    
    N = len(MPS_arr)                             # number of sites
    d = MPS_arr[0].shape[1]                      # local Hilbert space dimension
    qubits_per_site = int(np.ceil(np.log2(d)))   # number of qubits encoding local Hilbert space
    
    # Pre-processing of MPS: making sure all but the first tensor are left-normalized and normalizing MPS    
    for i in range(1,N):
        [Dleft, d, Dright] = MPS_arr[i].shape
        matrix_form = MPS_arr[i].reshape((Dleft*d,Dright))
        U, S, Vdag = np.linalg.svd(matrix_form, full_matrices=False)
        MPS_arr[i] = U.reshape((Dleft, d, int(U.size/(Dleft*d))))
        rest = np.diag(S).dot(Vdag)
        if i < N-1:
            MPS_arr[i+1] = np.tensordot(rest, MPS_arr[i+1], axes=([1],[0]))
        else:
            MPS_arr[0] = np.tensordot(rest, MPS_arr[0], axes=([1],[0]))

    norm = norm_MPS(MPS_arr, 'PBC')
    MPS_arr[0] = 1/norm * MPS_arr[0]

    # Generating matrix product disentanglers
    MPD_arr = MPD_generator_PBC(MPS_arr)

    # Generating quantum circuit
    [Dleft, d, Dright] = MPS_arr[0].shape
    num_extra_qubits = int(np.log2(MPD_arr[0].shape[0])) - int(mt.ceil(np.log2(d)))
    aux = max(int(mt.ceil(np.log2(d))) - int(mt.ceil(np.log2(Dleft))) - int(mt.ceil(np.log2(Dright))), 0)
    
    total_num_qubits = N*qubits_per_site + num_extra_qubits
    
    q = qiskit.QuantumRegister(total_num_qubits)
    qc = qiskit.QuantumCircuit(q)

    U_0 = np.transpose(np.conjugate(MPD_arr[N-1]))
    num_qubits_0 = int(np.log2(U_0.shape[0]))
    MPD0_gate = qiskit.quantum_info.operators.Operator(U_0)
    num_qubits_for_last_bond = int(mt.ceil(np.log2(MPS_arr[N-1].shape[2])))
    last_bond_register = q[total_num_qubits-1-aux-num_qubits_for_last_bond:total_num_qubits-1-aux]
    qc.unitary(MPD0_gate, last_bond_register+q[:num_qubits_0-num_qubits_for_last_bond])
        
    for i in range(1,N-1):
        U_i = np.transpose(np.conjugate(MPD_arr[N-1-i]))
        num_qubits_i = int(np.log2(U_i.shape[0]))
        MPDi_gate = qiskit.quantum_info.operators.Operator(U_i)
        qc.unitary(MPDi_gate, q[i*qubits_per_site:i*qubits_per_site+num_qubits_i])
    
    U_N = np.transpose(np.conjugate(MPD_arr[0]))
    num_qubits_N = int(np.log2(U_N.shape[0]))
    MPDN_gate = qiskit.quantum_info.operators.Operator(U_N)
    qc.unitary(MPDN_gate, q[total_num_qubits-num_qubits_N:])

    return qc

def Plesch_Brukner_preparation(psi, n, all_parts=False):
    """
    Function that implements Schmidt-decomposition-based method to
    prepare arbitrary statevectors, originally introduced in M. Plesch
    and C. Brukner, Phys. Rev. A 83, 032302 (2011). For a pedagogical
    description, see Sec. IX of B. Murta, P. M. Q. Cruz and J. Rossier,
    Phys. Rev. Research 5, 013190 (2023).
    
    This function applies to both even and odd numbers of qubits n. If 
    all_parts == True, subcircuits B, U and V are provided separately
    (see Fig. 5(a) of Phys. Rev. Research 5, 013190 (2023) for a
    clarification of the notation). Otherwise, the full circuit is given.
    Circuits are provided in Qiskit format.
    """
    
    if n == 1:
        qc = single_qubit_state_preparation(psi)
    else:
        M = np.reshape(psi,(2**(int(n/2) + n % 2),2**(int(n/2))))
        U, s, Vdagger = np.linalg.svd(M)

        U_circ = qiskit.quantum_info.operators.Operator(U)
        Vconj_circ = qiskit.quantum_info.operators.Operator(np.transpose(Vdagger))

        if int(n/2) == 1:
            B_circ = single_qubit_state_preparation(s).to_gate()
        else:
            B_circ = Plesch_Brukner_preparation(s, int(n/2)).to_gate()

        q = qiskit.QuantumRegister(int(n))
        qc = qiskit.QuantumCircuit(q)
        qc.append(B_circ, q[:int(n/2)])
        for i in range(int(n/2)):
            qc.cx(q[i],q[int(n/2)+i])

        qc.unitary(U_circ, q[int(n/2):])      # most significant qubits for U
        qc.unitary(Vconj_circ, q[:int(n/2)])  # least significant qubits for Vconj
        
    if all_parts:
        return B_circ, U_circ, Vconj_circ
    else:
        return qc
    
def Shende_Bullock_Markov_preparation(psi, n):
    """
    Generic state preparation method introduced in Sec. 4 (Theorem 9) of
    "Synthesis of Quantum Logic Circuits", V. V. Shende, S. S. Bullock and 
    I. L. Markov, arXiv:quant-ph/0406176v5. First part of function finds 
    parameters of multiplexed operations by disentangling one qubit at a
    time, thus arriving at the fiducial state. Second part generates the
    unitary that performs inverse of this process, i.e., the preparation
    of the input state psi, which is a n-qubit state. If explicit_decomp
    == True, Theorems 4 and 8 of the reference are used to perform the
    explicit basis gate decomposition of the MCRy and MCRz operations.
    Otherwise they are just left as high-level Qiskit operations. Outputs
    preparation circuit in Qiskit format.
    """
    
    # Finding the parameters (theta, phi) for every multiplexor in reversed order
    list_of_theta_arrays = []
    list_of_phi_arrays = []
    for m in reversed(range(1,n+1)):
        relevant_psi = psi[::2**(n-m)]
        full_U = np.zeros((2**m,2**m), dtype = np.cdouble)
        theta_array = []
        phi_array = []
        for i in range(int(2**(m-1))):
            new_psi = relevant_psi[2*i:2*(i+1)]
            new_qc, [new_theta, new_phi] = single_qubit_state_preparation(new_psi/np.linalg.norm(new_psi), params_out=True)
            theta_array.append(new_theta)
            phi_array.append(new_phi)
            new_U = ry_matrix(-new_theta).dot(rz_matrix(-new_phi))
            full_U[2*i:2*(i+1), 2*i:2*(i+1)] = new_U
        psi = np.kron(full_U, np.eye(2**(n-m))).dot(psi)
        list_of_theta_arrays.append(theta_array)
        list_of_phi_arrays.append(phi_array)
        
    list_of_theta_arrays.reverse()
    list_of_phi_arrays.reverse()

    # Generating quantum circuit with all multiplexors in right order for preparation
    qc = QuantumCircuit(n)
    
    for i in range(n):
        if i == 0:
            qc.ry(list_of_theta_arrays[0][0], n-1)
            qc.rz(list_of_phi_arrays[0][0],n-1)
        else:
            num_qubits = i+1
            list_of_qubits = list(range(n-i-1,n))
            new_multiplexor_y = multiplexor_basis_gate_decomp('y', list_of_theta_arrays[i], num_qubits, True, True, True)
            new_multiplexor_z = multiplexor_basis_gate_decomp('z', list_of_phi_arrays[i], num_qubits, True, True, False)
            qc.append(new_multiplexor_y, list_of_qubits)
            qc.append(new_multiplexor_z, list_of_qubits)

    return qc

def single_qubit_state_preparation(psi_1, params_out=False):
    """
    Function that generates quantum circuit to prepare a
    given single-qubit state starting from |0). Outputs 
    single-qubit circuit in Qiskit format.
    """
    
    theta = 2*np.arctan(np.absolute(psi_1[1])/np.absolute(psi_1[0]))
    phi = np.angle(psi_1[1]) - np.angle(psi_1[0])
    
    qc = qiskit.QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    
    if params_out:
        return qc, [theta, phi]        
    else:
        return qc    



