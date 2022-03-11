def horner_scheme(coefficients: list, x):
    """ Given a list of polynomial coefficients,
        i.e. [1, 0, 3, 4] -> 4x^3 + 3x^2 + 1
        Compute the polynomial value of a given point
    """
    if len(coefficients) == 1: return coefficients[0]
    return coefficients[0] + x * horner_scheme(coefficients[1:], x) 