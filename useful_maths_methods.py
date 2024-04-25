import numpy as np, scipy as sp, math as mt, cmath as cmt

def binary_rep(dec_num, num_bits):
    """
    Returns string corresponding to binary representation of
    positive integer dec_num with a total of num_bits bits.
    For example, for num_bits = 10 and dec_num = 61, the output
    is '0000111101'.
    """
    
    bin_num = bin(dec_num)[2:]
    while len(bin_num) < num_bits:
        bin_num = '0'+bin_num
    return bin_num

def direct_sum(A, B, complex_valued=True):
    if complex_valued:
        C = np.zeros(np.add(A.shape,B.shape), dtype=complex)
    else:
        C = np.zeros(np.add(A.shape,B.shape))
    C[:A.shape[0],:A.shape[1]] = A
    C[A.shape[0]:,A.shape[1]:] = B
    
    return C

def fidelity(psi, phi):
    """
    Returns fidelity (i.e., modulus squared of overlap) between
    two states psi and phi with the same dimensionality. The input
    states need not be vectors; the numpy.ndarry flatten method is
    employed just in case. Inputs must be numpy arrays.
    """
    
    return np.absolute(np.dot(np.conjugate(psi.flatten()), phi.flatten()))**2

def line_intersection(line1, line2, range=[-mt.inf,mt.inf]):
    """
    Function that determines if two lines intersect. Each line
    input is a tuple of tuples, where each tuple within it contains
    the (x,y) coordinates of a point along which the line passes
    through. E.g., line1 = ((0,0), (2,1)) for a line that passes
    through points (0,0) and (2,1). If range is provided, then
    only intersections with y coordinate within the range are
    considered. If there is no intersection, the output is False,
    otherwise the output is the x and y coordinates of the
    intersection.
    """
    
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    if y > range[0] and y < range[1]:
        return x, y
    else:
        return False
    
def local_operator_onto_vector(operator, vector, active_indices):
    """
    Function that computes action of given local operator
    onto a given vector by summing only over the relevant
    indices instead of padding operator with identities
    to make it have the same linear dimension as the vector.
    The vector is already provided as a tensor with indices
    of the right dimensionality. active_indices is a list
    of the indices on which the operator acts nontrivially.
    Of course, the product of the dimensions of those indices
    must equal the linear dimension of the operator. The
    operator is returned in the same form as it was provided,
    with the indices ordered in exactly the same way.
    """
    
    dim = operator.shape[1]
    
    # Placing active indices in the most significant positions
    original_shape = vector.shape
    inactive_indices = []
    for i in range(len(vector.shape)):
        if i not in active_indices:
            inactive_indices.append(i)
    vector = np.transpose(vector, active_indices+inactive_indices)
    
    # Applying local operator onto vector
    op_dot_vec = np.outer(operator, vector)
    op_dot_vec = np.reshape(op_dot_vec, (dim, dim, dim, vector.size//dim))
    op_dot_vec = np.einsum(op_dot_vec, [0,1,1,2], [0,2])
    op_dot_vec = np.reshape(op_dot_vec, original_shape)
    
    # Reordering indices to recover original order
    op_dot_vec = np.transpose(op_dot_vec, np.argsort(active_indices+inactive_indices))
    
    return op_dot_vec

def normalization(psi):
    """
    Function that normalizes given state psi. Outputs normalized
    state, as well as norm prior to normalization.
    """
    
    norm = np.linalg.norm(psi)
    psi = 1/norm * psi
    
    return psi, norm
    
def Pauli_matrices_1Q():
    """
    Function that returns list of single-qubit Pauli matrices
    [I, X, Y, Z], including the identity.
    """
    
    I = np.eye(2)
    X = np.array([[0,1],
                  [1,0]])
    Y = np.array([[0,-1j],
                  [1j, 0]])
    Z = np.array([[1, 0],
                  [0,-1]])
    
    return [I,X,Y,Z]

def rx_matrix(theta):
    """
    Returns matrix representation of Rx matrix according to
    Mike & Ike definition (10th Anniversary, p. 174, Eq. (4.4)).
    """
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2),  np.cos(theta/2)]])

def ry_matrix(theta):
    """
    Returns matrix representation of Ry matrix according to
    Mike & Ike definition (10th Anniversary, p. 174, Eq. (4.5)).
    """
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]])

def rz_matrix(phi):
    """
    Returns matrix representation of Rz matrix according to
    Mike & Ike definition (10th Anniversary, p. 174, Eq. (4.6)).
    """
    return np.array([[np.exp(-1j*phi/2),                0],
                     [                0, np.exp(1j*phi/2)]])

def Walsh_Hadamard_trans_matrix(n, no_prefactor=False):
    """
    Function that returns 2^n x 2^n matrix representation of
    n-qubit Walsh-Hadamard transform. If no_prefactor == True,
    the factor of 1/sqrt(2) is not included in each Hadamard gate. 
    """
    if no_prefactor:
        prefactor = 1
    else:
        prefactor = 1/np.sqrt(2)
    Hadamard = prefactor*np.array([[1, 1],
                                   [1,-1]])
    WH_matrix = Hadamard
    for i in range(n-1):
        WH_matrix = np.kron(Hadamard, WH_matrix)
    
    return WH_matrix
