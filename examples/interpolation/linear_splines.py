import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt 
from numan.polynomials import LinearSpline
from numan.data import Point

if __name__ == '__main__': 

    points = [ 
        Point(1, 3), 
        Point(2, 6),  
        Point(3, 9), 
        Point(4, 5), 
        Point(5, 2), 
        Point(6, 3), 
        Point(7, 8), 
        Point(8, 6), 
        Point(9, 2) 
    ]
    
    lspline = LinearSpline(points)
    
    draw_x = np.linspace(1, 9, 300)
    draw_y = np.array( [ lspline(x) for x in draw_x ] )

    plt.figure(figsize=(8,8))
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    
    sns.lineplot(x=draw_x, y=draw_y, estimator=None)

    draw_pts_x = [ p.node  for p in points ]
    draw_pts_y = [ p.value for p in points ]
    plt.scatter(draw_pts_x, draw_pts_y, color='none', edgecolors='purple')

    plt.show() 

