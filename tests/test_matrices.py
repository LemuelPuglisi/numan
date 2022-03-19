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