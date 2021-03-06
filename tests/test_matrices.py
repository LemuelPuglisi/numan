import numpy as np

from numan import matrices as mtx 

def test_random_normal_matrix():
    A = mtx.generate_random_matrix((2,2))
    r, c = A.shape
    assert r == 2 and c == 2


def test_random_symmetric_matrix():
    A = mtx.generate_symmetric_matrix(3)
    assert np.array_equal(A, A.transpose()) 


def test_is_symmetric():
    A = mtx.generate_symmetric_matrix(10)
    assert mtx.is_symmetric(A)


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


def test_is_upper_triangular_matrix():
    A = np.array([
        [1, 1, 1], 
        [0, 1, 1], 
        [0, 0, 1]
    ])
    B = np.array([
        [1, 1, 1], 
        [0, 1, 1], 
        [2, 0, 1]
    ])
    assert mtx.is_upper_triangular_matrix(A)
    assert not mtx.is_upper_triangular_matrix(B)


def test_is_lower_triangular_matrix():
    A = np.array([
        [1, 0, 0],
        [1, 1, 0], 
        [1, 1, 1]
    ])
    B = np.array([
        [1, 1, 1], 
        [0, 1, 1], 
        [2, 0, 1]
    ])
    assert mtx.is_lower_triangular_matrix(A)
    assert not mtx.is_lower_triangular_matrix(B)


def test_random_tridiagonal_matrix():
    A = mtx.generate_random_tridiagonal_matrix(5)
    for i in range(5): 
        for j in range(5):
            if j not in [i-1, i, i+1]: assert A[i,j] == 0


def test_is_tridiagonal():
    At = np.array([
        [1, 1, 0, 0], 
        [1, 1, 1, 0], 
        [0, 1, 1, 1], 
        [0, 0, 1, 1]
    ])
    Ant = np.array([
        [1, 1, 0, 3], 
        [1, 1, 1, 0], 
        [0, 1, 1, 1], 
        [3, 0, 1, 1]
    ])
    B = np.array([
        [2, 2], 
        [3, 3]
    ])
    assert mtx.is_tridiagonal(At)
    assert not mtx.is_tridiagonal(Ant)
    assert mtx.is_tridiagonal(B)


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
    assert mtx.is_positive_definite(Ap)
    

def test_is_positive_definite_matrix_with_non_positive_matrix():
    An = np.array([
        [3, 5], 
        [7, 1]
    ])
    assert not mtx.is_positive_definite(An)


def test_random_positive_definite_matrix():
    A = mtx.generate_random_positive_definite_matrix(4)
    assert mtx.is_positive_definite(A)


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


def test_is_strictly_diagonally_dominant():
    A = mtx.generate_random_strictly_diagonally_dominant_matrix(3)
    B = np.ones((3, 3))
    assert mtx.is_strictly_diagonally_dominant(A)
    assert not mtx.is_strictly_diagonally_dominant(B)


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


def test_cofactor_matrix(): 
    A = np.array([
        [1, 2], 
        [3, 4]        
    ])
    C = mtx.cofactor_matrix(A)
    DI = np.round(np.matmul(A, C.transpose()))
    assert DI[0, 0] == -2 and DI[1, 0] == 0 and DI[1, 0] == 0 and DI[1, 1] == -2  


def test_algebraic_complement():
    A = np.array([
        [ 1, 3, -1], 
        [ 2, 4,  0], 
        [-1, 2,  2]
    ])
    assert round(mtx.determinant(A)) == -12
    assert round(mtx.determinant(A), 1) == round(mtx.determinant(A), 2)
    assert round(mtx.determinant(A), 2) == round(mtx.determinant(A), 3)


def test_determinant():
    A1 = np.array([
        [ 1, 3, -1], 
        [ 2, 4,  0], 
        [-1, 2,  2]
    ])
    A2 = np.array([
        [1, 2], 
        [3, 4]        
    ])
    assert round(np.linalg.det(A1)) == round(mtx.determinant(A1))
    assert round(np.linalg.det(A2)) == round(mtx.determinant(A2))


def test_matrix_inverse(): 
    A = np.array([
        [1, 2], 
        [3, 4]        
    ])
    exp_Ai = np.array([
        [-2, 1], 
        [1.5, 0.5]
    ])
    obt_Ai = mtx.inverse(A)
    np.allclose(exp_Ai, obt_Ai)


def test_binet_theorem():
    A = np.array([
        [ 1, 3], 
        [-2, 1]
    ])
    B = np.array([
        [ 3, 2], 
        [ 1, 0]
    ])
    det_AB = mtx.determinant(np.matmul(A, B))
    assert mtx.binet_theorem(A, B, det_AB)


def test_permutation_matrix(): 
    eP = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    oP = mtx.create_permutation_matrix(0, 2, 3)
    assert np.array_equal(eP, oP)    