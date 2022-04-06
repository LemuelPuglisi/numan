import numpy as np

from numan import linearsystems as ls 

def test_forward_substitution():
    A = np.array([
        [3, 0, 0],
        [4, 1, 0], 
        [5, 2, 6], 
    ])
    b = np.array(
        [6, 8, 2]
    )
    x = ls._forward_substitution(A, b)
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
    x = ls._backward_substitution(A, b)
    assert np.array_equal(x, np.array([-4/3, 0, 2]))


def test_gem():
    A = np.array([
        [1,  2, 3], 
        [2,  1, 4], 
        [3, -3, 1]
    ])
    b = np.array([ 1., 2., 1. ])
    x = ls.gem(A, b)
    ob = A.dot(x)
    assert np.allclose(b, ob)


def test_cholesky_solve():
    A = np.array([
        [   4,  12, -16],
        [  12,  37, -43],
        [ -16, -43,  98]
    ])
    b = np.array([ 2, 12, 30 ])
    xo = ls.cholensky_solve(A, b)
    xe = np.array([ -11/18, 14/9, 8/9 ])
    assert np.allclose(xo, xe)
