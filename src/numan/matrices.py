import numpy as np

# from enum import Enum


# class MatrixType(Enum):
#     """ Enum containing matrix types. 
#     """
#     NORMAL              = 0
#     SQUARE              = 1
#     HERMITIAN           = 2
#     SYMMETRIC           = 3
#     IDENTITY            = 4
#     DIAGONAL            = 5
#     UPPER_TRIANGULAR    = 6
#     LOWER_TRIANGULAR    = 7
#     TRIDIAGONAL         = 8


#     HESSEMBERG          = 9
#     POSITIVE            = 10
#     NEGATIVE            = 11
#     POSITIVE_SEMI_DEFINITE = 12
#     NEGATIVE_SEMI_DEFINITE = 13
#     STRICT_DIAGONALLY_DOMINANT = 14
#     WEAKLY_DIAGONALLY_DOMINANT = 15


def _rme():
    """ Returns a random matrix element between 1 and 10.
    """
    return np.random.randint(low=1, high=10)


def matrix_shape(func):
    def inner(shape: tuple):
        assert len(shape) == 2
        return func(shape)
    return inner


@matrix_shape
def generate_random_matrix(shape: tuple):
    """ Generate a random matrix. 
    """
    r, c = shape
    return np.random.randint(10, size=(r, c))


@matrix_shape
def generate_symmetric_matrix(shape: tuple):
    """ Generate a random symmetric matrix.
    """
    r, c = shape
    A = np.ndarray(shape)
    for i in range(r):
        for j in range(i, c):
            A[i, j] = A[j, i] = _rme()
    return A
    

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


def generate_random_upper_triangular_matrix(n):
    """ Generate a n x n random upper triangular matrix 
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i,j] = _rme()
    return A


def generate_random_upper_triangular_matrix(n):
    """ Generate a n x n random upper triangular matrix 
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            A[i,j] = _rme()
    return A


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