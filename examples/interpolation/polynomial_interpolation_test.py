import numpy as np
import seaborn as sns

from numan import interpolation as itp
from matplotlib import pyplot as plt 

if __name__ == '__main__':

    f = np.vectorize(lambda x: np.sin(x))
    x = np.linspace(0, 2*np.pi, 100)
    y_f = f(x)

    plt.figure(figsize=(8,8))

    legend_labels = []

    for nn in [4, 6, 8]:
        nodes = np.linspace(0, 2*np.pi, nn)
        funcs = f(nodes)
        points = [ itp.Point(n, f) for  n, f in zip(nodes, funcs)]
        
        # First method
        # p = itp.indeterminate_coefficients_method(points)
        # y_p = [ p.evaluate(_x) for _x in x ]
        
        # Second method
        # y_p = [ itp.lagrange_polynomials_method(points, _x) for _x in x ]

        # third method
        p = itp.newton_polynomial(points)
        y_p = [ p.evaluate(_x) for _x in x ]

        sns.lineplot(x=x, y=y_p, estimator=None, linewidth=3)
        sns.scatterplot(x=nodes, y=funcs)
        legend_labels.append(f'p(x)  - degree {nn-1}')
        legend_labels.append(f'nodes - degree {nn-1}')

    # equiscaled axys.
    plt.xlim(0, 2*np.pi)
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal', adjustable='box')

    sns.lineplot(x=x, y=y_f, estimator=None, linewidth=2)
    legend_labels.append('sin(x)')

    plt.legend(legend_labels)
    plt.show()