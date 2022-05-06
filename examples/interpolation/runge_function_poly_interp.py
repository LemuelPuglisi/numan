import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt 
from numan import interpolation as itp

if __name__ == '__main__':

    rungef = lambda x: 1 / (1 +  x**2)    
    f = np.vectorize(rungef)
    x = np.linspace(-10, 10, 200)
    y_f = f(x)
    plt.figure(figsize=(8,8))

    legend_labels = []

    for nn in [14]:        
        
        nodes = np.linspace(-5, 5, nn)
        funcs = f(nodes)
        points = [ itp.Point(n, f) for  n, f in zip(nodes, funcs)]
        x = np.linspace(-10, 10, 200)
    
        # First method
        # p = itp.indeterminate_coefficients_method(points)
        # y_p = [ p.evaluate(_x) for _x in x ]
        
        # Second method
        # y_p = [ itp.lagrange_polynomials_method(points, _x) for _x in x ]

        # third method
        p = itp.newton_polynomial(points)
        y_p = [ p.evaluate(_x) for _x in x ]

        sns.lineplot(x=x, y=y_p, estimator=None, linewidth=3)
        sns.scatterplot(x=nodes, y=funcs, s=100)
        legend_labels.append(f'p(x)  - degree {nn-1}')
        legend_labels.append(f'nodes - degree {nn-1}')

    """ Interpolation through Chebyshev node distribution. """
    cnodes = np.polynomial.chebyshev.chebpts1(14)
    cnodes = np.array([ c * 5 for c in cnodes ])
    funcs = f(cnodes)
    points = [ itp.Point(n, f) for  n, f in zip(cnodes, funcs)]
    cp = itp.newton_polynomial(points)
    x = np.linspace(-10, 10, 200)
    y_p = [ cp.evaluate(_x) for _x in x ]
    sns.lineplot(x=x, y=y_p, estimator=None, linewidth=3)
    sns.scatterplot(x=cnodes, y=funcs, s=100)
    legend_labels.append(f'cp(x)  - degree {nn-1}')
    legend_labels.append(f'cnodes - degree {nn-1}')

    """ Plot the Runge function. 
    """
    # equiscaled axys.
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    sns.lineplot(x=x, y=y_f, estimator=None, linewidth=2)
    #legend_labels.append('sin(x)')
    legend_labels.append('runge function')
    plt.legend(legend_labels)
    plt.show()