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


# def p_correct_significant_digits(real, mesure, p):
#     """ Test if the first p significant digits corresponds in
#         the two mesurements. 
#     """
#     return relative_error(real, mesure) < ( 10 ** (1 - p) )

if __name__ == '__main__': 

    # test 1
    x = 992
    y = 1001
    assert absolute_error(x, y) == 9

    # test 2
    x = 12500
    y = 13000
    assert relative_error(x, y) == 0.04

    # test 3
    x = 624.428731
    y = 624.428711
    assert p_correct_decimals(x, y, 4)

    # test 4 //  7 cifre sign
    x = 624.428731
    y = 624.428711
    
    # p_correct_significant_digits(x, y, 5)
    # p_correct_significant_digits(x, y, 6)
    # p_correct_significant_digits(x, y, 7)
    # p_correct_significant_digits(x, y, 8)
    # assert p_correct_significant_digits(x, y, 9)
    # assert not p_correct_significant_digits(x, y, 10)
    r = relative_error(x, y)
    print(r)