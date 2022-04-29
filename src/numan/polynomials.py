from tracemalloc import StatisticDiff
import numpy as np
import timeit

from abc import ABC, abstractmethod
from typing import List

#--------------------------------------------------#

class Polynomial(ABC):
    """ Polynomial Abstract class.
    """

    def __init__(self, degree):
        """ Return a polynomial instance.
        """
        self.degree = degree


    def __call__(self, x: np.ScalarType):
        """ Call the evaluate method directly.
        """
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x: np.ScalarType):
        """ Evaluate the scalar x using the
            polynomial.
        """
        pass


#--------------------------------------------------#


class StandardPolynomial(Polynomial):

    def __init__(self, coefficients: List):
        degree = len(coefficients) - 1
        super().__init__(degree) 
        self.coefficients = coefficients

    def evaluate(self, x: np.ScalarType):    
        return self._horner_scheme(self.coefficients, x)

    def _horner_scheme(self, coefficients, x):
        if len(coefficients) == 1: return coefficients[0]
        return coefficients[0] + x * self._horner_scheme(coefficients[1:], x) 


#--------------------------------------------------#


class NddPolynomial(Polynomial):

    def __init__(self, bterms: List, nodes: List):
        super().__init__(len(bterms)) 
        self.bterms = bterms
        self.nodes  = nodes


    def evaluate(self, x: np.ScalarType):
        production = lambda x, i: np.prod([ (x - self.nodes[j]) for j in range(i) ])
        x = sum([ b * production(x, i) for i, b in enumerate(self.bterms) ])
        return x


#--------------------------------------------------#


def from_coefficients(coefficients: np.array):
    return StandardPolynomial(coefficients)



if __name__ == '__main__':
    pass
    # p = NddPolynomial([-1, 3, 1], [0, 1, 2])
    # print(p(3))
