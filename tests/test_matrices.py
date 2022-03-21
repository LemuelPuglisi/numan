import numpy as np

from numan import matrices as mtx 

def test_random_normal_matrix():
    A = mtx.generate_random_matrix((2,2))
    r, c = A.shape
    assert r == 2 and c == 2


def test_random_symmetric_matrix():
    A = mtx.generate_symmetric_matrix((3, 3))
    assert np.array_equal(A, A.transpose()) 


def test_identity_matrix():
    I = mtx.generate_identity_matrix(4)
    A = mtx.generate_random_matrix((4, 4))
    assert np.array_equal(np.matmul(I, A), A)  


def test_random_diagonal_matrix():
    A = mtx.generate_random_diagonal_matrix(2)
    assert A[1,0] == 0 and A[0,1] == 0 


def test_random_upper_triangular_matrix():
    A = mtx.generate_random_upper_triangular_matrix(3)
    assert A[1, 0] == 0 and A[2, 0] == 0 and A[2, 1] == 0


def test_random_lower_triangular_matrix():
    A = mtx.generate_random_lower_triangular_matrix(3)
    assert A[0, 1] == 0 and A[0, 2] == 0 and A[1, 2] == 0 


def test_random_tridiagonal_matrix():
    A = mtx.generate_random_tridiagonal_matrix(5)
    for i in range(5): 
        for j in range(5):
            if j not in [i-1, i, i+1]: assert A[i,j] == 0


def test_random_hessemberg_matrix():
    A = mtx.generate_random_hessemberg_matrix(5)
    for i in range(5): 
        for j in range(5):
            if j > i + 1 or i > j +1: assert A[i,j] == 0


def test_is_positive_definite_matrix_with_positive_matrix():
    Ap = np.array([
        [ 2, -1,  0], 
        [-1,  2, -1], 
        [ 0, -1,  2]
    ])
    assert mtx.is_positive_definite_matrix(Ap)
    

def test_is_positive_definite_matrix_with_non_positive_matrix():
    An = np.array([
        [3, 5], 
        [7, 1]
    ])
    assert not mtx.is_positive_definite_matrix(An)


def test_random_positive_definite_matrix():
    A = mtx.generate_random_positive_definite_matrix(4)
    assert mtx.is_positive_definite_matrix(A)


def test_is_positive_semidefinite_matrix_with_positive_matrix():
    Ap = np.array([
        [ 2, -1,  0], 
        [-1,  2, -1], 
        [ 0, -1,  2]
    ])
    assert mtx.is_positive_semidefinite_matrix(Ap)
    

def test_is_positive_semidefinite_matrix_with_non_positive_matrix():
    An = np.array([
        [3, 5], 
        [7, 1]
    ])
    assert not mtx.is_positive_semidefinite_matrix(An)


def test_random_positive_definite_matrix():
    A = mtx.generate_random_positive_semidefinite_matrix(4)
    assert mtx.is_positive_semidefinite_matrix(A)


def test_random_strictly_diagonally_dominant_matrix():
    A = mtx.generate_random_strictly_diagonally_dominant_matrix(4)
    for i in range(4):
        ds = sum( [ A[i, j] for j in range(4) if j != i ] )
        assert A[i,i] > ds


def test_random_weakly_diagonally_dominant_matrix():
    A = mtx.generate_random_weakly_diagonally_dominant_matrix(4)
    for i in range(4):
        ds = sum( [ A[i, j] for j in range(4) if j != i ] )
        assert A[i,i] >= ds


def test_cofactor():
    A = np.array([
        [ 1, 3, -1], 
        [ 2, 4,  0], 
        [-1, 2,  2]
    ])
    assert round(mtx.cofactor(A, 0, 0)) == 8    
    assert round(mtx.cofactor(A, 0, 1)) == -4


def test_algebraic_complement():
    A = np.array([
        [ 1, 3, -1], 
        [ 2, 4,  0], 
        [-1, 2,  2]
    ])
    assert round(mtx.determinant(A)) == -12
    assert round(mtx.determinant(A), 1) == round(mtx.determinant(A), 2)
    assert round(mtx.determinant(A), 2) == round(mtx.determinant(A), 3)


def test_matrix_inverse(): 
    pass