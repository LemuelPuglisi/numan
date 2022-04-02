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