import numpy as np

from numan import linearsystems as ls 
from numan import matrices as mx

def test_forward_substitution():
    A = np.array([
        [3, 0, 0],
        [4, 1, 0], 
        [5, 2, 6], 
    ])
    b = np.array(
        [6, 8, 2]
    )
    S = ls.LinearSystem(A, b)
    x = ls._forward_substitution(S)
    assert np.array_equal(x, np.array([2, 0, -4/3]))


def test_backward_substitution():
    A = np.array([
        [6, 2, 5],
        [0, 1, 4], 
        [0, 0, 3], 
    ])
    b = np.array(
        [2, 8, 6]
    )
    S = ls.LinearSystem(A, b)
    x = ls._backward_substitution(S)
    assert np.array_equal(x, np.array([-4/3, 0, 2]))


def test_gem_solve():
    A = np.array([
        [1,  2, 3], 
        [2,  1, 4], 
        [3, -3, 1]
    ])
    b = np.array([ 1., 2., 1. ])
    S = ls.LinearSystem(A, b)
    x = ls.gem_solve(S)
    ob = A.dot(x)
    assert np.allclose(b, ob)


def test_cholesky_solve():
    A = np.array([
        [   4,  12, -16],
        [  12,  37, -43],
        [ -16, -43,  98]
    ])
    b = np.array([ 2, 12, 30 ])
    S = ls.LinearSystem(A, b)
    xo = ls.cholensky_solve(S)
    xe = np.array([ -11/18, 14/9, 8/9 ])
    assert np.allclose(xo, xe)


def test_thomas_solve():
    A = np.array([
        [2., 1., 0.],
        [1., 2., 1.],
        [0., 1., 2.],
    ])
    b = np.array([ 3, 2, 1 ])
    S = ls.LinearSystem(A, b)
    xo = ls.thomas_solve(S)
    xe = np.array([ 1.5, 0, 0.5 ])
    assert np.allclose(xo, xe)


def test_jacobi_solve():
    # the convergence of the method is ensured by diagonally
    # dominant linear systems.
    A = mx.generate_random_strictly_diagonally_dominant_matrix(5)
    b = np.random.rand(5)
    S1 = ls.LinearSystem(A, b) # we build two because gem will modify the system
    S2 = ls.LinearSystem(A, b)
    xo = ls.jacobi_solve(S1, max_iter=1000, eps=1e-7)
    xe = ls.gem_solve(S2)
    assert np.allclose(xo, xe, atol=1e-05)


def test_gauss_seidel_solve():
    # the convergence of the method is ensured by diagonally
    # dominant linear systems.
    A = mx.generate_random_strictly_diagonally_dominant_matrix(5)
    b = np.random.rand(5)
    S1 = ls.LinearSystem(A, b) # we build two because gem will modify the system
    S2 = ls.LinearSystem(A, b)
    xo = ls.gauss_seidel_solve(S1, max_iter=1000, eps=1e-7)
    xe = ls.gem_solve(S2)
    assert np.allclose(xo, xe, atol=1e-05)


def test_sor_solve():
    # the convergence of the method is ensured by diagonally
    # dominant linear systems.
    A = mx.generate_random_strictly_diagonally_dominant_matrix(5)
    b = np.random.rand(5)
    S1 = ls.LinearSystem(A, b) # we build two because gem will modify the system
    S2 = ls.LinearSystem(A, b)
    xo = ls.sor_solve(S1, max_iter=1000, eps=1e-7)
    xe = ls.gem_solve(S2)
    assert np.allclose(xo, xe, atol=1e-05)


def test_gradient_solve():
    # the convergence of the method is ensured by diagonally
    # dominant linear systems.
    A = mx.generate_random_positive_definite_matrix(3)
    b = np.random.rand(3)
    S1 = ls.LinearSystem(A, b) # we build two because gem will modify the system
    S2 = ls.LinearSystem(A, b)
    xo = ls.gradient_solve(S1, max_iter=1000)
    xe = ls.gem_solve(S2)
    assert np.allclose(xo, xe, atol=1e-05)


def test_conjugate_gradient_solve():
    # the convergence of the method is ensured by diagonally
    # dominant linear systems.
    A = mx.generate_random_positive_definite_matrix(3)
    b = np.random.rand(3)
    S1 = ls.LinearSystem(A, b) # we build two because gem will modify the system
    S2 = ls.LinearSystem(A, b)
    xo = ls.conjugate_gradient_solve(S1, max_iter=1000)
    xe = ls.gem_solve(S2)
    assert np.allclose(xo, xe, atol=1e-05)
