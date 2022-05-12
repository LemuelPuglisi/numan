import numpy as np

from typing import List
from numan import matrices as mx
from numan import linearsystems as ls
from numan import polynomials as poly

class Point:
    def __init__(self, node, value, grad=None):
        self.node   = node
        self.value  = value
        self.grad   = grad

    def set_grad(self, grad):
        self.grad = grad



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


def osculatory_interpolation(points): 
    """ Return a polynomial that (almost) interpolates the points,
        using the osculatory interpolation (Hermitian Interpolation).
        Recall the theorem on osculatory interpolation where if 
        there are n+1 data points and their first derivative, then
        there is only one 2n+1 grade polynomial p(x) such that
        p(xi) = f(xi) and p'(xi) = f'(xi). 
    """
    for p in points:
        assert p.grad is not None, "Osculatory interpolation requires first derivative"

    n = len(points)         # n + 1
    m = 2*n                 # 2n + 2 
    matrix_rows = []
    b = []
    for p in points:
        
        # normal equality 
        row_grad_0 = [ p.node ** i for i in range(m) ]
        matrix_rows.append(row_grad_0)
        b.append(p.value)

        # derivative equality 
        row_grad_1 = [ (i+1) * (p.node ** i) for i in range(m-1) ]
        row_grad_1.insert(0, 0) # first coefficient is zero. 
        matrix_rows.append(row_grad_1)
        b.append(p.grad)

    A, b = np.array(matrix_rows), np.array(b)
    print(A.shape)
    Ls = ls.LinearSystem(A, b)
    return poly.from_coefficients(ls.gem_solve(Ls))


def chebichev_zeros(n, min=-1, max=1):
    """ returns the zeros of the n-th chebychev polynomial
        scaled to the [min, max] range. 
    """
    ith_zero = lambda i: np.cos(((i + .5) / n) * np.pi)
    zeros = [ ith_zero(i) for i in range(n-1, -1, -1) ] 
    scale = lambda x: (max-min) * ((x+1)/2) + min
    scaled_zeros = [ scale(x) for x in zeros ]
    return np.array(scaled_zeros)


if __name__ == '__main__':
    pass