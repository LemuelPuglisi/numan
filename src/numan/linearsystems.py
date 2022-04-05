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
    """ Perform the Gaussian elimination method.  
    """
    assert len(A.shape) == 2 and len(b.shape) == 1
    assert A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]    
    assert mx.determinant(A) != 0
    # assert can_apply_gem(A)
    L, U, P = mx.LU(A)
    # b = np.matmul(P, b)
    y =  _forward_substitution(L, b)
    x = _backward_substitution(U, y)
    return x


def can_apply_gem(A: np.ndarray):
    """ If all the principal minors of the matrix
        have a non-zero determinant, all the diagonals
        elements will be non-zero during the Gaussian 
        elimination method. 
    """
    pms = mx.get_matrix_principal_minors(A)
    for pm in pms: 
        if mx.determinant(pm) == 0: 
            return False
    return True


if __name__ == '__main__': 
    pass
    # A = np.array([
    #     [ 1, -1,  1], 
    #     [-6,  1, -1],
    #     [ 3,  1,  1]
    # ])
    # b = np.array([ 2, 3, 4 ])
    # x = gem(A, b)
    # print("Actual: ", x)
    # print("Expected: ", [ -1, 2, 5 ])
