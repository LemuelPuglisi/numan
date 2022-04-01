import numpy as np

def spectral_radius(A: np.ndarray):
    """ Calculate the spectral radius of a matrix, that
        is the greater eigenvalue of the matrix.
    """
    pass 

    
def gerschgorin_disks(A: np.ndarray):
    """ Given a n x n matrix, this function returns n 
        Gerschgorin disks as pairs (<center>, <radius>), 
        recalling that the disks are defined in the 
        complex plan. The Gerschgorin disk theorem ensure
        that all the eigenvalues are inside the disks, where
        each disk is a set defined as follows:  
        
        { z complex : |z - center| <= radius }

    """
    n, m = A.shape
    assert n == m 
    disks = []
    for i in range(n):
        center = A[i, i]
        radius = sum([ np.abs(A[i,j]) for j in range(m) if i != j ])
        disks.append( (center, radius) )
    return disks