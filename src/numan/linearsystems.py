import numpy as np
import timeit

from scipy.linalg import hilbert
from numan import matrices as mx


class LinearSystem:
    """ Define a linear system
    """
    def __init__(self, A: np.ndarray, b:np.ndarray):
        assert len(A.shape) == 2, "A must be a matrix."
        assert len(b.shape) == 1, "b must be a vector."
        assert A.shape[0] == b.shape[0], "A rows and b must have the same size"
        self.A = A.astype(np.double)
        self.b = b.astype(np.double)
        self.equations = A.shape[0]
        self.variables = A.shape[1]
    
    def is_squared(self):
        return self.A.shape[0] == self.A.shape[1]



def _forward_substitution(S: LinearSystem):
    """ The forward substitution method take a lower 
        triangular matrix A as input (coefficient matrix
        of a linear system) and a vector of costant terms b. 
        The function computes the solution of the linear system. 
    """
    assert S.is_squared(), "Needs a squared linear system."
    x = np.zeros(S.variables)
    for i in range(S.equations):
        tmp = S.b[i]
        # Given the index i=3, the range returns the 
        # previous elements (j=0, j=1, j=2). When i=0, 
        # we should set range(0) that is empty. 
        for j in range( i if i - 1 >= 0 else 0 ): tmp -= S.A[i, j] * x[j]
        x[i] = tmp / S.A[i, i]
    return x


def _backward_substitution(S: LinearSystem):
    """ The backward substitution method take an upper 
        triangular matrix A as input (coefficient matrix
        of a linear system) and a vector of costant terms b. 
        The function computes the solution of the linear system. 
    """
    assert S.is_squared(), "Needs a squared linear system."
    n = S.equations
    x = np.zeros(S.variables)
    for i in reversed(range(n)):
        tmp = S.b[i]
        # The row iterations are reversed, so we start from 
        # the bottom of the matrix. We should also iterate 
        # the columns in a reversed manner (recall we need
        # to compute the solution components in a give order). 
        for j in range(n-1, i, -1): tmp -= S.A[i, j] * x[j]
        x[i] = tmp / S.A[i, i]
    return x


def gem(S: LinearSystem):
    """ Perform the Gaussian elimination method.  
    """
    assert S.is_squared(), "Needs a squared linear system."
    assert mx.determinant(S.A) != 0
    # assert can_apply_gem(A)
    L, U, P = mx.LU(S.A)
    # b = np.matmul(P, b)
    y =  _forward_substitution(LinearSystem(L, S.b))
    x = _backward_substitution(LinearSystem(U, y))
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


def cholensky_solve(S: LinearSystem):
    """ This function takes a positive-defined symmetric 
        matrix and a coefficient vector and solves the linear
        system using the Cholesky decomposition. 
    """
    assert mx.determinant(S.A) != 0
    L = mx.cholesky_decomposition(S.A)
    y =  _forward_substitution(LinearSystem(L, S.b))
    x = _backward_substitution(LinearSystem(L.T, y))
    return x


def thomas_solve(S: LinearSystem):
    """
        Thomas algorithm solves in linear time a tridiagonal
        system of equation. An example is: 

        a1 [b1 c1 00]
           [a2 b2 c2]
           [00 a3 b3] c3
        
        where a1 = c3 = 0. 

           [d1 d2 d3] coefficient vector.

        Reference: https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """    
    assert mx.is_tridiagonal(S.A)
    n, m = S.A.shape
    # this lambdas take an index as input and 
    # return a triple <value>, <row>, <column>
    # and work as mapping functions. 
    b_to_mat = lambda i: (S.A[i, i], i, i)
    a_to_mat = lambda i: (S.A[i, i-1], i, i-1)
    c_to_mat = lambda i: (S.A[i, i+1], i, i+1)

    for i in range(1, n):
        a, ar, ac = a_to_mat(i)
        b, br, bc = b_to_mat(i)
        bp, bpr, bpc = b_to_mat(i-1)
        cp, cpr, cpc = c_to_mat(i-1)
        _m = a / bp
        S.A[br, bc] = b - (_m * cp)
        S.b[i] = S.b[i] - (_m * S.b[i-1])

    x = np.zeros(m, dtype=np.float64)
    bn, _, _ = b_to_mat(n-1)
    x[m-1] = S.b[n-1] / bn

    for i in range(n-2, -1, -1):
        c, _, _ = c_to_mat(i)        
        b, _, _ = b_to_mat(i)
        x[i] = (S.b[i] - (c * x[i+1])) / b

    return x


if __name__ == '__main__': 

    A = np.array([
        [2., 1., 0.],
        [1., 2., 1.],
        [0., 1., 2.],
    ])

    b = np.array([ 3, 2, 1 ])

    x = thomas_solve(A, b)
    print(x)
    # print(r)

    # A = np.array([
    #     [ 1, -1,  1], 
    #     [-6,  1, -1],
    #     [ 3,  1,  1]
    # ])

    # A = hilbert(10)
    # # b = np.array([ 2, 3, 4 ])
    # b = np.random.random(10)
    # xo = gem(A, b)
    # xe = np.linalg.solve(A, b)

    # ea = sum(np.abs(xo - xe))
    # print(f"error = {ea}")

    # es = np.linalg.norm(xo - xe)
    # print(f"squared error: {es}")

    # # e = np.linalg.norm(xo - xe)
    # # print(f"error = {e}")

    # print("Actual: ", xo)
    # print("Expected: ", np.linalg.solve(A, b) )

    # t1 = timeit.timeit(lambda: gem(A, b), number=1)
    # t2 = timeit.timeit(lambda: np.linalg.solve(A, b), number=1)
    # print(f"t1: {t1}")
    # print(f"t2: {t2}")

