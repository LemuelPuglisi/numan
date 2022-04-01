import numpy as np

from numan import eigenvalues as ev

def test_gerschgorin_disks():
    A = np.array([
        [ 4, 3,  2], 
        [-3, 2, -1], 
        [-2, 1,  5]
    ])
    disks = ev.gerschgorin_disks(A)
    assert disks[0][0] == 4 and disks[0][1] == 5
    assert disks[1][0] == 2 and disks[1][1] == 4
    assert disks[2][0] == 5 and disks[2][1] == 3
