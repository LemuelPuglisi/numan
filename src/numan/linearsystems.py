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
        print(f"i: {i}")
        tmp = b[i]
        # The row iterations are reversed, so we start from 
        # the bottom of the matrix. We should also iterate 
        # the columns in a reversed manner (recall we need
        # to compute the solution components in a give order). 
        for j in range(n-1, i, -1): tmp -= A[i, j] * x[j]
        x[i] = tmp / A[i, i]
    return x