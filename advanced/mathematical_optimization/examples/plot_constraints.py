"""
Optimization with constraints
"""
import numpy as np
import pylab as pl
from scipy import optimize

x, y = np.mgrid[-2.9:5.8:.05, -2.5:5:.05]
x = x.T
y = y.T

for i in (1, 2):
    # Create 2 figure: only the second one will have the optimization
    # path
    pl.figure(i, figsize=(3, 2.5))
    pl.clf()
    pl.axes([0, 0, 1, 1])

    contours = pl.contour(np.sqrt((x - 3)**2 + (y - 2)**2),
                        extent=[-3, 6, -2.5, 5],
                        cmap=pl.cm.gnuplot)
    pl.clabel(contours,
            inline=1,
            fmt='%1.1f',
            fontsize=14)
    pl.plot([-1.5, -1.5,  1.5,  1.5, -1.5],
            [-1.5,  1.5,  1.5, -1.5, -1.5], 'k', linewidth=2)
    pl.fill_between([ -1.5,  1.5],
                    [ -1.5, -1.5],
                    [  1.5,  1.5],
                    color='.8')
    pl.axvline(0, color='k')
    pl.axhline(0, color='k')

    pl.text(-.9, 4.4, '$x_2$', size=20)
    pl.text(5.6, -.6, '$x_1$', size=20)
    pl.axis('equal')
    pl.axis('off')

# And now plot the optimization path
accumulator = list()

def f(x):
    # Store the list of function calls
    accumulator.append(x)
    return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)


# We don't use the gradient, as with the gradient, L-BFGS is too fast,
# and finds the optimum without showing us a pretty path
def f_prime(x):
    r = np.sqrt((x[0] - 3)**2 + (x[0] - 2)**2)
    return np.array(((x[0] - 3)/r, (x[0] - 2)/r))

optimize.fmin_l_bfgs_b(f, np.array([0, 0]), approx_grad=1,
                       bounds=((-1.5, 1.5), (-1.5, 1.5)))

accumulated = np.array(accumulator)
pl.plot(accumulated[:, 0], accumulated[:, 1])

pl.show()

