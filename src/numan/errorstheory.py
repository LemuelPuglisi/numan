import numpy as np

def absolute_error(real, mesure):
    """ Mesure the goodness of the approximation
        by giving the exact error between the 2 mesurements.  
    """
    return np.abs(real - mesure)


def relative_error(real, mesure):
    """ Mesure the goodness of the approximation
        by giving the percentage of error
    """
    assert real != 0
    return np.abs((real - mesure) / real) 


def p_correct_decimals(real, mesure, p):
    """ Test if the first p decimals corresponds in 
        the two mesurements. 
    """
    return absolute_error(real, mesure) < ( 10 ** (-p) )

#
# Machine discretization functions 
#

