from tkinter import E
import numpy as np

from enum import Enum


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

class Approximations(Enum):
    """ Enum containing the approximation methods. 
    """
    CHOPPING = 1
    ROUNDING = 2


class MachineNumberSet:
    """ This class expose the characteristics of a machine number 
        set, using the floating point mechanism. So, given a base b, 
        the number of bits t assigned to the significand, a lower bound
        L and and upper bound U to the exponent, we build the machine
        number set.   
    """

    epsilon_fmap = { 
        Approximations.CHOPPING: lambda b, t: b ** (- t + 1), 
        Approximations.ROUNDING: lambda b, t: b ** (- t + 1) * .5 
    }


    def __init__(self, b, t, U, L, approx = Approximations.ROUNDING):
        self.base   = b
        self.bits   = t
        self.lower  = L
        self.upper  = U
        self.approx = approx


    def cardinality(self):
        """ Since the machine number set is a discrete, finite set, 
            we can calculate precisely its dimension as it follows. 
        """
        a = self.base ** (self.bits - 1)
        r = self.upper - self.lower + 1
        return 1 + 2 * (self.base - 1) * a * r


    def smallest(self):
        """ Returns the smallest number rapresentable.
        """
        return self.base ** (self.lower - 1)


    def largest(self):
        """ Returns the largest number rapresentable.
        """
        return (self.base ** self.upper) * (1 - self.base ** (-self.bits))


    def machine_epsilon(self, approx=Approximations.ROUNDING):
        """ Returns the machine epsilon
        """
        assert self.approx in Approximations
        f = self.epsilon_fmap[approx]
        return f(self.base, self.bits)
