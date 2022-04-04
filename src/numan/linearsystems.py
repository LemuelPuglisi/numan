import numpy as np

from numan import matrices as mx

def _forward_substitution(A: np.ndarray, b: np.ndarray):
    """ The forward substitution method take a lower 
        triangular matrix A as input (coefficient matrix
        of a linear system) and a vector of costant terms b. 
        The function computes the solution of the linear system. 
    """
    assert len(A.shape) == 2 and len(b.shape) == 1
    assert A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]    
    n, _ = A.shape
    x = np.zeros(n)
    for i in range(n):
        tmp = b[i]
        # Given the index i=3, the range returns the 
        # previous elements (j=0, j=1, j=2). When i=0, 
        # we should set range(0) that is empty. 
        for j in range( i if i - 1 >= 0 else 0 ): tmp -= A[i, j] * x[j]
        x[i] = tmp / A[i, i]
    return x


def _backward_substitution(A: np.ndarray, b: np.ndarray):
    """ The backward substitution method take an upper 
        triangular matrix A as input (coefficient matrix
        of a linear system) and a vector of costant terms b. 
        The function computes the solution of the linear system. 
    """
    assert len(A.shape) == 2 and len(b.shape) == 1
    assert A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]
    n, _ = A.shape
    x = np.zeros(n)
    for i in reversed(range(n)):
        tmp = b[i]
        # The row iterations are reversed, so we start from 
        # the bottom of the matrix. We should also iterate 
        # the columns in a reversed manner (recall we need
        # to compute the solution components in a give order). 
        for j in range(n-1, i, -1): tmp -= A[i, j] * x[j]
        x[i] = tmp / A[i, i]
    return x


def gem(A: np.ndarray, b: np.ndarray):
    """ Gaussian elimination method.  
    """
    assert len(A.shape) == 2 and len(b.shape) == 1
    assert A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]    
    assert mx.determinant(A) != 0
    n, _ = A.shape
    # We will iteratively build the L~ matrix 
    # multipling L_i x Lt at each i-th iteration 
    Lt = mx.generate_identity_matrix(n)    
    # The U matrix is equivalent to the A matrix at 
    # step n-1 of the gaussian elimination method, since
    # A^(n-1) = L_{n-1} x L_{n-2} x ... x L_2 x L_1 x A. 
    U = A.copy()
    for i in range(n-1): 
        Li = mx.generate_identity_matrix(n)
        for j in range(i+1, n): 
            Li[j, i]  = - U[j, i] / U[i, i]     # Computing the multipliers.         
        U = np.matmul(Li, U)                    # Updating the A matrix (will define U)
        Lt = np.matmul(Li, Lt)                  # Updating the L~ matrix
    L = np.linalg.inv(Lt)               # TODO: ask if _L inverse matrix is computed
                                        # in a particular way, since we're using numpy. 
    y =  _forward_substitution(L, b)    # Computing Ly=b
    x = _backward_substitution(U, y)    # Computing Ux=y
    return x


if __name__ == '__main__': 
    pass
