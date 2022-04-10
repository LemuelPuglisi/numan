from logging import warning
import numpy as np

from numan import matrices as mx


class LinearSystem:
    """ Define a linear system.
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


def gem_solve(S: LinearSystem):
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
    # forward elimination
    for i in range(1, n):
        a, ar, ac = a_to_mat(i)
        b, br, bc = b_to_mat(i)
        bp, bpr, bpc = b_to_mat(i-1)
        cp, cpr, cpc = c_to_mat(i-1)
        _m = a / bp
        S.A[br, bc] = b - (_m * cp)
        S.b[i] = S.b[i] - (_m * S.b[i-1])
    # backward substitution
    x = np.zeros(m, dtype=np.float64)
    bn, _, _ = b_to_mat(n-1)
    x[m-1] = S.b[n-1] / bn
    for i in range(n-2, -1, -1):
        c, _, _ = c_to_mat(i)
        b, _, _ = b_to_mat(i)
        x[i] = (S.b[i] - (c * x[i+1])) / b
    return x


def jacobi_solve(S: LinearSystem, max_iter=100, eps=1e-8):
    """ Solves the system of linear equations using the Jacobi 
        iterative method. 
    """
    if not mx.is_strictly_diagonally_dominant(S.A):
        warning(("Matrix not strictly diagonally dominant, "
                    "method may not converge."))
    x = np.random.rand(S.variables)
    # The termination criterion is based on measuring the relative change between
    # the previous and the current solution vector x.
    change_ratio = lambda xa, xb: np.linalg.norm(xa - xb) / np.linalg.norm(xa)
    for _ in range(max_iter):
        prev_x = x.copy()
        for i in range(S.variables):
            rowsum = sum([ S.A[i, j] * prev_x[j] for j in range(S.variables) if i != j ])
            x[i] = (S.b[i] - rowsum) / S.A[i,i]
        if change_ratio(prev_x, x) < eps:
            break
    return x


def change_ratio(xa, xb):
    """ ratio of change between two vectors """ 
    return np.linalg.norm(xa - xb) / np.linalg.norm(xa)



def gauss_seidel_solve(S: LinearSystem, max_iter=100, eps=1e-8):
    """ Solves the system of linear equations using the Gauss-Seidel 
        iterative method. 
    """
    if not mx.is_strictly_diagonally_dominant(S.A) \
        and not (mx.is_symmetric(S.A) and mx.is_positive_definite(S.A)):
        warning(("Matrix not strictly diagonally dominant, "
                    "nor symmetric diagonally positive definite, " 
                    "method may not converge."))
    x = np.random.rand(S.variables)
    # The termination criterion is based on measuring the relative change between
    # the previous and the current solution vector x.
    for _ in range(max_iter):
        prev_x = x.copy()
        for i in range(S.variables):
            rowsum = sum([ S.A[i, j] * x[j] for j in range(S.variables) if i != j ])
            x[i] = (S.b[i] - rowsum) / S.A[i,i]
        if change_ratio(prev_x, x) < eps:
            break
    return x


def sor_solve(S: LinearSystem, max_iter=100, eps=1e-8, omega=.9):
    """ Solves the system of linear equations using the SOR  
        (Successive Over-Relaxation) iterative method. 
    """
    assert omega > 0 and omega < 2, "omega should be in ]0, 2[ to avoid method divergence."
    if not mx.is_strictly_diagonally_dominant(S.A) \
        and not (mx.is_symmetric(S.A) and mx.is_positive_definite(S.A)):
        warning(("Matrix not strictly diagonally dominant, "
                    "nor symmetric diagonally positive definite, " 
                    "method may not converge."))
    x = np.random.rand(S.variables)
    # The termination criterion is based on measuring the relative change between
    # the previous and the current solution vector x.
    for _ in range(max_iter):
        prev_x = x.copy()
        for i in range(S.variables):
            rowsum = sum([ S.A[i, j] * x[j] for j in range(S.variables) if i != j ])
            x[i] = (S.b[i] - rowsum) / S.A[i,i]
        # after an iteration of gauss-seidel, we take a fraction omega
        # from the new x, and a fraction 1-omega of the old x. 
        x = omega * (x) + (1 - omega) * prev_x
        if change_ratio(prev_x, x) < eps:
            break
    return x


def gradient_solve(S: LinearSystem, max_iter=100, eps=1e-8):
    """ Solve the system of linear equation reducing it
        to an optimization problem, where it finds the minimum 
        of the quadratic form of the system. 
    """
    if not mx.is_positive_definite(S.A):
        warning(("Matrix not definite positive, method may not converge."))
    # Empirically, method works even with non-symmetric matrices. 
    # if not mx.is_symmetric(S.A):
    #     S = LinearSystem(np.matmul(S.A.T, S.A), S.b)        # analogous problem when A is not symmetric 

    A, b = S.A, S.b
    x = np.random.rand(S.variables)

    for k in range(max_iter):
        prev_x = x.copy()
        dk = np.matmul(A, x) - b                            # calculating the gradient. 
        rk = - dk                                           # direction and rate of fastest decrease
        denominator = np.matmul(rk.T, A).dot(rk)            # denominator to calculate alpha
        if denominator == 0: 
            warning('Avoiding zero division, stopping the method.')
            break
        ak = (rk.T.dot(rk)) / denominator                   # alpha = length of the step
        x = x + ak * rk                                     # calculating the new solution 
        if change_ratio(prev_x, x) < eps:
            break
    return x


def conjugate_gradient_solve(S: LinearSystem, max_iter=100, eps=1e-8):
    """ Solve the linear system using the conjugate gradient method. 
    """  
    assert mx.is_symmetric(S.A) \
        and mx.is_positive_definite(S.A), "A must be symmetric definite positive"
    A, b = S.A, S.b
    # Since A is positive definite, it induces a scalar product
    # <.,.>_A : R^n x R^n -> R, defined for each u,v vectors of R^n
    # as u^T * A * v (alwais real due to the posive definition of A).
    # A pair (u, v) such that <u,v>_A = 0 is called A-conjugate.  
    Adot = lambda u, v: (u.T @ A) @ v
    # The solution x of Ax=b corresponds to the minimum point
    # of its the quadratic form. 
    x_prev = np.random.rand(S.variables)
    r_prev = b - (A @ x_prev)
    p_prev = r_prev
    for _ in range(max_iter): 
        adot_prev = Adot(p_prev, p_prev) # used many times. 
        if adot_prev == 0: 
            warning('Avoiding zero division, stopping the method.')
            break
        alpha = (p_prev.T @ r_prev) / adot_prev
        x = x_prev + (alpha * p_prev)
        r = b - (A @ x)
        beta = Adot(p_prev, r) / adot_prev
        p = r - (beta * p_prev)
        if change_ratio(x_prev, x) < eps:
            break
        x_prev = x
        r_prev = r
        p_prev = p
    return x


if __name__ == '__main__':
    pass

    # A = np.array([
    #     [30,  8,  9], 
    #     [8,  30,  8],
    #     [9,   8, 30] 
    # ])

    # b = np.array([5, 4, 5])

    # S1 = LinearSystem(A, b)
    # S2 = LinearSystem(A, b)
    
    # xe = gem_solve(S2)
    # xo = conjugate_gradient_solve(S1)
    # print(xo)
    # print(xe)
