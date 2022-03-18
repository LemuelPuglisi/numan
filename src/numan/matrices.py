
from enum import Enum


class MatrixType(Enum):
    """ Enum containing matrix types. 
    """
    NORMAL              = 0
    SQUARE              = 1
    HERMITIAN           = 2
    SYMMETRIC           = 3
    IDENTITY            = 4
    DIAGONAL            = 5
    UPPER_TRIANGULAR    = 6
    LOWER_TRIANGULAR    = 7
    TRIDIAGONAL         = 8
    HESSEMBERG          = 9
    POSITIVE            = 10
    NEGATIVE            = 11
    POSITIVE_SEMI_DEFINITE = 12
    NEGATIVE_SEMI_DEFINITE = 13
    STRICT_DIAGONALLY_DOMINANT = 14
    WEAKLY_DIAGONALLY_DOMINANT = 15



def generate_matrix(size: tuple, type=MatrixType.NORMAL): 
    pass