import numpy as np

from abc import ABC, abstractmethod
from typing import List
from numan.data import Point

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


class Spline(ABC):
    """ Abstract class for splines. 
    """

    def __init__(self, points: List[Point]):
        self.points = points
        self.calculate_coefficients()


    def __call__(self, x: np.ScalarType):
        """ Call the evaluate method directly.
        """
        return self.evaluate(x)


    @abstractmethod
    def calculate_coefficients(self):
        """ Calculate the spline polynomial coefficients.
        """
        pass


    @abstractmethod
    def evaluate(self, x: np.ScalarType):
        """ Evaluate the scalar x using the spline.
        """
        pass


#--------------------------------------------------#


class LinearSpline(Spline):

    def calculate_coefficients(self):
        """ Instead of calculating the coefficients, we 
            can evaluate the function directly using the points. 
        """
        nodes = [ point.node for point in self.points ]
        self.argmin = min(nodes)
        self.argmax = max(nodes)
        

    def evaluate(self, x: np.ScalarType):
        assert x >= self.argmin and x <= self.argmax, "Point outside spline range."
        
        value = 0
        for i, point in enumerate(self.points):
            xc = point.node
            yc = point.value

            if i > 0:
                xp = self.points[i-1].node
                if x >= xp and x <= xc:
                    value += ((x - xp) / (xc - xp)) * yc

            if i < (len(self.points) - 1):
                xn = self.points[i+1].node
                if x >= xc and x <= xn:
                    value += ((xn - x) / (xn - xc)) * yc

        return value

#--------------------------------------------------#


def from_coefficients(coefficients: np.array):
    return StandardPolynomial(coefficients)



if __name__ == '__main__':

    points = [ Point(1, 5), Point(2, 3), Point(4, 6), Point(3, 5) ]

    spline = LinearSpline(points)

    print(spline(3.2))