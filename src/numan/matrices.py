import numpy as np
import itertools

def _rme():
    """ Returns a random matrix element between 1 and 10.
    """
    return np.random.randint(low=1, high=10)


def matrix_shape(func):
    def inner(shape: tuple):
        assert len(shape) == 2
        return func(shape)
    return inner


def is_matrix(func):
    def inner(A: np.ndarray):
        assert len(A.shape) == 2
        return func(A)
    return inner


@matrix_shape
def generate_random_matrix(shape: tuple):
    """ Generate a random matrix. 
    """
    r, c = shape
    return np.random.randint(10, size=(r, c))


def generate_symmetric_matrix(n):
    """ Generate a random symmetric matrix.
    """
    A = np.ndarray((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i, j] = A[j, i] = _rme()
    return A
    

@is_matrix
def is_symmetric(A: np.ndarray):
    assert A.shape[0] == A.shape[1]
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j] != A[j, i]: return False
    return True 


def generate_identity_matrix(n):
    """ Generate a n x n identity matrix 
    """
    return np.identity(n)


def generate_random_diagonal_matrix(n):
    """ Generate a n x n random diagonal matrix
    """
    A = np.zeros((n, n))
    for i in range(n): A[i,i] = _rme()
    return A


@is_matrix
def is_upper_triangular_matrix(A: np.ndarray):
    """ Return True if the matrix is upper triangular """
    n, _ = A.shape
    for i in range(1, n):
        for j in range(0, i):
            if A[i, j] != 0: return False
    return True


def generate_random_upper_triangular_matrix(n):
    """ Generate a n x n random upper triangular matrix 
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i,j] = _rme()
    return A


@is_matrix
def is_lower_triangular_matrix(A: np.ndarray):
    """ Return True if the matrix is lower triangular """
    n, _ = A.shape
    for i in range(0, n):
        for j in range(i+1, n):
            print((i, j))
            if A[i, j] != 0: return False
    return True


def generate_random_lower_triangular_matrix(n):
    """ Generate a n x n random lower triangular matrix 
    """
    A = generate_random_upper_triangular_matrix(n)
    return A.transpose()


def generate_random_tridiagonal_matrix(n):
    """ Generate a n x n random tridiagonal matrix
    """
    A = np.zeros((n, n))
    for i in range(n): 
        A[i,i] = _rme()
        if (i - 1 >= 0): A[i, i - 1] = _rme()
        if (i + 1 < n):  A[i, i + 1] = _rme()
    return A


@is_matrix
def is_tridiagonal(A: np.ndarray):
    """ Returns True if A is a tridiagonal matrix """
    n, m = A.shape
    if n < 3 or m < 3: return True
    for i in range(n):
        for j in range(m):
            if i == j or i == j - 1 or i == j + 1: continue
            if A[i, j] != 0: return False
    return True 
            

def generate_random_hessemberg_matrix(n):
    """ Generate a n x n random Hessemberg matrix. 
        The (ij) element of an Hessemberg matrix is 0
        if j > i + 1 or if i > j + 1.
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j <= i + 1 and i <= j + 1:
                A[i,j] = _rme()
    return A


def is_positive_definite(A: np.ndarray):
    """ This method uses the Sylvester theorem to check 
        if the matrix is positive definite. The Sylvester 
        theorem says: 
        matrix A is positive definite if and only if det(Ak)>0
        for k = 1, ..., n. Where Ak is the matrix formed by the
        first k row and k columns of A. 
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    r, _ = A.shape 
    for i in range(r):
        k = i + 1
        Ak = A[:k, :k]
        if np.linalg.det(Ak) <= 0: return False
    return True


def generate_random_positive_definite_matrix(n):
    """ This method generates a (pseudo) random n x n positive matrix.
        The following property states that: 
        if A is symmetric, diagonally dominant and has a positive diagonal, 
        then A is a positive definite matrix. 
    """
    A = generate_symmetric_matrix((n, n))
    for i in range(n): A[i, i] = sum( [ A[i, j] for j in range(n) if j != i ] )
    return A


def get_matrix_principal_minors(A: np.ndarray):
    """ This function extracts all the principal minors from the
        matrix. A principal minor is a minor where the index of 
        the deleted column corresponds with the index of the 
        deleted row. 
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    n, _ = A.shape
    principal_minors = []
    for current_minors_size in range(1, n + 1):
        rows_cols_to_delete = n - current_minors_size
        combinations = itertools.combinations(range(n), rows_cols_to_delete)
        combinations = [ list(combination) for combination in combinations ]
        for combination in combinations:
            combination.sort(reverse=True) # so the delete func doesn't mess indices 
            B = np.copy(A)
            for i in combination:
                B = _remove_ith_col_row(B, i)
            principal_minors.append(B)
    return principal_minors


def _remove_ith_col_row(A, i):
    A = np.delete(A, i, 0)
    A = np.delete(A, i, 1)
    return A


def is_positive_semidefinite_matrix(A: np.ndarray):
    """ An Sylvester analogous theorem holds for characterizing 
        positive semidefinite hermitian matrices, except that it 
        is no longer sufficient to consider only the leading principal
        minors: 
        An Hermitian matrix M is positive-semidefinite if and only if
        all principal minors of M are nonnegative. 
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    principal_minors = get_matrix_principal_minors(A)
    return all( [ np.linalg.det(B) >= 0 for B in principal_minors ] )
    

def generate_random_positive_semidefinite_matrix(n):
    """ This function generate a positive semidefinite matrix and uses
        the following observation:
        A^T * A is positive semidefinite. 
        proof: let x be a non-zero column vector. Then we have: 
        (1.)  x^T * A^T * A * x = 
        (2.) = (x^T * A^T) * A * x = 
        (3.) = (A * x)^T * A * x 
        Notice that Ax is also a non-zero column vecotr, so (3.) is the 
        square of the inner product of Ax. Then:
        (A * x)^T * A * x = ||Ax||^2 >= 0. 
    """
    A = generate_random_matrix((n, n))
    return np.matmul(A.transpose(), A)


def generate_random_strictly_diagonally_dominant_matrix(n):
    """ Generate a n x n strictly diagonally dominant random matrix
    """
    A = generate_random_matrix((n, n))
    for i in range(n): A[i, i] = sum( [ A[i, j] for j in range(n) if j != i ] ) + 1
    return A


@is_matrix
def is_strictly_diagonally_dominant(A: np.ndarray):
    """ Returns True if the matrix is strictly diagonally dominant """
    assert A.shape[0] == A.shape[1], "Square matrix required."
    n = A.shape[0]
    for i in range(n): 
        aii = A[i, i]  
        row = sum( [ A[i, j] for j in range(n) if j != i ] )
        if aii <= row: return False
    return True


def generate_random_weakly_diagonally_dominant_matrix(n):
    """ Generate a n x n weakly diagonally dominant random matrix
    """
    A = generate_random_matrix((n, n))
    for i in range(n): A[i, i] = sum( [ A[i, j] for j in range(n) if j != i ] )
    return A


def cofactor(A: np.ndarray, row: int, col: int):
    """ Compute the (row,col) matrix element cofactor. 
    """
    Arc = np.delete(A,   row, 0)
    Arc = np.delete(Arc, col, 1)
    s = 1 if (row + col) % 2 == 0 else -1
    return s * np.linalg.det(Arc)


@is_matrix
def cofactor_matrix(A: np.ndarray):
    """ Compute the cofactor matrix """
    n, m = A.shape
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = cofactor(A, i, j)
    return C


def determinant(A: np.ndarray, row=0):
    """ Compute the matrix determinant using the Laplace theorem. 
        The theorem states that: 
        Given the arbitrary i-th row, the determinant is equivalent 
        to the sum of the products of the A(ij) element, times the 
        A(ij) cofactor. 
    """
    assert A.shape[0] == A.shape[1]
    return sum ( A[row, j] * cofactor(A, row, j) for j in range(A.shape[1]) )


def inverse(A: np.ndarray):
    """ Compute the inverse matrix. 
        Given the transposed cofactor matrix C, since 
        AC = CA = det(A) x I => (C/det(A))A = I 
        hence A^-1 = C / det(A) is the inverse matrix of A.
    """
    assert A.shape[0] == A.shape[1]
    # we are using cofactor without swapping (i,j) indices
    # so we need to transpose the matrix. The swap produces
    # directly the transpose. 
    return cofactor_matrix(A).transpose() / determinant(A)
    

def binet_theorem(A: np.ndarray, B: np.ndarray, det: float):
    """ From the Binet theorem, we know that: 
        det(AB) = det(A) x det(B)
        So this function returns True if the determinant det
        is equal to det(A) x det(B)
    """
    assert len(A.shape) == 2 and len(B.shape) == 2
    det_ab = determinant(A) * determinant(B)
    return round(det, 8) == round(det_ab, 8)


def create_permutation_matrix(i: int, j: int, n: int): 
    """ Create a n x n permutation matrix P. 
        Pre-multipling  P x A will swap rows i and j.
        Post-multipling A x P will swap cols i and j.  
    """
    assert i >= 0 and i < n and j >= 0 and j < n
    P = generate_identity_matrix(n)
    P[i, i] = P[j, j] = 0
    P[i, j] = P[j, i] = 1
    return P

@is_matrix
def LU(A: np.ndarray):
    """ This function calculate the LU factorization of
        the matrix A, moreover: A = L x U. A third value 
        P is returned along with L and U: that's the permutation
        matrix in order to reproduce the partial pivot operations. 
        (You may want to apply those steps to a coefficient vector b
        while solving a linear system, see function gem @ linearsystem.py)
    """
    assert determinant(A) != 0
    n, _ = A.shape
    Lt = P = generate_identity_matrix(n)    
    U = A.copy()
    for i in range(n-1):
        Li = generate_identity_matrix(n)
        for j in range(i+1, n): 
            Li[j, i]  = - U[j, i] / U[i, i]              
        U = np.matmul(Li, U)    
        Lt = np.matmul(Li, Lt)
    L = inverse(Lt)
    # P is just an identity matrix (for now)
    return L, U, P


# def partial_pivot(A: np.ndarray, c: int):
#     """ Search the greater element in the specified column (c) 
#         and swap the row containing that element with the row c. 
#         We should map this permutation even in the coefficient 
#         vector b. 
#     """
#     idx = c + np.argmax(np.abs(A[c:, c]))
#     P = create_permutation_matrix(c, idx, A.shape[0])
#     Ap = np.matmul(P, A)
#     return Ap, P


@is_matrix
def cholesky_decomposition(A: np.ndarray):
    """ This function takes a positive-definite symmetric matrix
        as input and returns the Cholesky decomposition matrix. 
    """
    assert is_symmetric(A)
    assert is_positive_definite(A)
    n, _ = A.shape
    L = np.zeros((n, n))
    L[0, 0] = np.sqrt(A[0, 0])
    for c in range(n):
        for r in range(c, n):
            L[r, c] = A[r, c] - sum([ L[r, k] * L[c, k] for k in range(c)])
            L[r, c] = np.sqrt(L[r, c]) if r == c else L[r, c] / L[c, c] 
    return L