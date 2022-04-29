import numpy as np

from typing import List
from numan import matrices as mx
from numan import linearsystems as ls
from numan import polynomials as poly

class Point:
    def __init__(self, node, value):
        self.node = node
        self.value = value


def indeterminate_coefficients_method(points: List[Point]):
    """ Apply the Indeterminate Coefficients Method to
        find the polynomial p(x) that interpolates the 
        function.
    """
    n = len(points)
    A = [ [p.node ** i for i in range(n)  ]  for p in points ]
    b = [ p.value for p in points ]
    A, b = np.array(A), np.array(b)
    Ls = ls.LinearSystem(A, b)
    return poly.from_coefficients(ls.gem_solve(Ls))


def lagrange_polynomials_method(points: List[Point], x: np.ScalarType): 
    """ Apply the Finite Difference Method to find the 
        polynomial p(x) that interpolates the function. 
    """
    n = len(points)
    value = 0
    for i, point in enumerate(points):
        Li_x = 1
        n_skip_i = [ j for j in range(n) if j != i ]
        for j in n_skip_i:
            Li_x *= (x - points[j].node) / (points[i].node - points[j].node)
        value += point.value * Li_x
    return value


def newton_polynomial(points: List[Point]):
    """ Apply the Newton Divided Differences method
        to calculate the Newton polynomial that fits 
        the points. 
    """
    nodes = [ point.node for point in points  ]
    funcs = [ point.value for point in points ]
    n = len(nodes)
    # build the pyramid structure as a list of list. 
    dd_pyr = []
    dd_pyr.append(funcs)
    #-----------------------------------------------------------------
    # The algorithm uses an index game to fill the 
    # structure with correct divided differences (dd).
    # With (i) we recall the layer of the pyramid, and
    # with (j) we indicate the j-th element of the layer. 
    # The j-th dd of layer (i) is computed using the following formula: 
    # dd[i,j] = (dd[i-1, j] - dd[i-1, j+1]) / (x[j+i] - x[j])   
    #-----------------------------------------------------------------
    for i in range(1, n):
        pl = i-1
        dd_pyr.append([])
        for j in range(0, n - i):
            pdd = dd_pyr[pl][j]
            ndd = dd_pyr[pl][j+1]
            px = nodes[j]
            nx = nodes[j+i]
            dd = (ndd-pdd)/(nx-px)
            dd_pyr[i].append(dd)
    bterms = [ l[0] for l in dd_pyr ]
    return poly.NddPolynomial(bterms, nodes) 


if __name__ == '__main__':
    pass

    # nodes = [-2, -1, 0, 1, 2]
    # funcs = [15, -2, -5, -6, 7]
    # points = [ Point(n, v) for n, v in zip(nodes, funcs) ]
    # p = newton_polynomial(points)
    # print(p.evaluate(0))
    # print(p.evaluate(1))
    # print(p.evaluate(2))
    # print(p.evaluate(3))