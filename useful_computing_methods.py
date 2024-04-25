import numpy as np, scipy as sp, math as mt, cmath as cmt, random
import qiskit, qiskit_aer as Aer
from qiskit import *

def depth_and_CNOT_count_qcircuit(qc, output_circuit = False, 
                                  ref_gates=['u3', 'cx'], op_level=3):
    """
    Returns circuit depth (including both CNOTs and 1Q gates) and CNOT count
    of an input quantum circuit qc. ref_gates corresponds to the basis_gates
    input and op_level is the optimization_level to be provided to the Qiskit
    transpile method.
    """
    
    qc_transpile = transpile(qc,basis_gates=ref_gates,optimization_level=op_level)
    
    circuit_depth = qc_transpile.depth()
    if 'cx' in qc_transpile.count_ops().keys():
        cnot_count = qc_transpile.count_ops()['cx']
    else:
        cnot_count = 0
    
    if output_circuit:
        return [circuit_depth, cnot_count], qc_transpile
    else:
        return [circuit_depth, cnot_count]
    
def random_state_generator(n, seed=None):
    """
    Function that outputs the 2^n-dimensional vector corresponding
    to a random n-qubit wave function.
    """
    psi = qiskit.quantum_info.random_statevector(2**n, seed).data
        
    return psi

def random_unitary_generator(n, seed=None):
    """
    Function that outputs the 2^n x 2^n matrix representation
    of a random n-qubit unitary operation.
    """
    U = qiskit.quantum_info.random_unitary(2**n, seed).data
    
    return U

def statevector_output_qcircuit(qc, gauge=False):
    """
    Function that returns output of n-qubit circuit as a
    2^n-dimensional wave function. If gauge == True, a 
    global phase factor is applied to ensure the first
    entry is a real number.
    """

    backend = Aer.StatevectorSimulator()
    qc_t = transpile(qc, backend)
    job = backend.run(qc_t).result()
    outputstate = job.data()['statevector'].data
    
    if gauge:
        outputstate = 1/np.exp(1j*np.angle(outputstate[0]))*outputstate
    
    return outputstate

def sorting_list_according_to_order_of_another_list(A,B):
    """
    Function that takes as inputs two lists A and B
    and sorts list A according to order of list B.
    """
    
    return [x for _, x in sorted(zip(B, A))]

def unitary_rep_qcircuit(qc, gauge=False):
    """
    Function that returns unitary matrix representation
    of quantum circuit. If gauge == True, a global phase 
    factor is applied to ensure the (0,0) entry is real.
    """
    backend = Aer.UnitarySimulator()
    qc_t = transpile(qc, backend)
    job = backend.run(qc_t).result()
    unitary = job.data()['unitary'].data
    
    if gauge:
        unitary = 1/np.exp(1j*np.angle(unitary[0,0]))*unitary
    
    return unitary
