from numan import interpolation as itp
from numan import polynomials as pl
from typing import Callable

def generate_equispaced_points():
    nodes = [ 1., 5., 9. ]
    funcs = [ 3., 9., 6. ]
    return [ itp.Point(x, y) for x, y in zip(nodes, funcs) ]


def interpolates(evaluate: Callable, points):
    equalities = [ evaluate(p.node) == p.value for p in points ]
    return all(equalities)


def test_indeterminate_coefficients_method():
    epts = generate_equispaced_points()
    px = itp.indeterminate_coefficients_method(epts)
    evaluate = lambda x: px(x)
    assert interpolates(evaluate, epts)


def test_lagrange_polynomials_method():
    epts = generate_equispaced_points()
    evaluate = lambda x: itp.lagrange_polynomials_method(epts, x)
    assert interpolates(evaluate, epts)


def test_newton_polynomial():
    epts = generate_equispaced_points()
    px = itp.newton_polynomial(epts)
    evaluate = lambda x: px(x)
    assert interpolates(evaluate, epts)
