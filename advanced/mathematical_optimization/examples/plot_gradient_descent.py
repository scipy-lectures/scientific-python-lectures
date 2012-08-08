"""
Demo gradient descent
"""
import numpy as np
import pylab as pl
from scipy import optimize

x_min, x_max = -1, 2
y_min, y_max = 2.25/3*x_min - .2, 2.25/3*x_max - .2

###############################################################################
# A formatter to print values on contours
def super_fmt(value):
    if value > 1:
        if np.abs(int(value) - value) < .1:
            out = '$10^{%.1i}$' % value
        else:
            out = '$10^{%.1f}$' % value
    else:
        value = np.exp(value - .01)
        if value > .1:
            out = '%1.1f' % value
        elif value > .01:
            out = '%.2f' % value
        else:
            out = '%.2e' % value
    return out

###############################################################################
# Super ill-conditionned problem
def rosenbrock(x):
    x, y = x
    x = 4*x + 1
    y = 4*y + 3
    return .5*(1 - x)**2 + (y - x**2)**2


def rosenbrock_prime(x):
    x, y = x
    x = 4*x + 1
    y = 4*y + 3
    return 4*np.array((-2*.5*(1 - x) - 4*x*(y - x**2), 2*(y - x**2)))


def rosenbrock_hessian(x):
    x, y = x
    x = 4*x + 1
    y = 4*y + 3
    return 4*4*np.array((
                    (1 - 4*y + 12*x**2, -4*x),
                    (             -4*x,    2),
                   ))


###############################################################################
# Gaussian and quadratic functions with varying conditionning

def gaussian(x):
    return np.exp(-x**2)


def gaussian_prime(x):
    return -2*x*np.exp(-x**2)


def gaussian_prime_prime(x):
    return -2*np.exp(-x**2) + 4*x**2*np.exp(-x**2)


def mk_gauss(epsilon):
    def f(x):
        return -gaussian(.5*np.sqrt(x[0]**2 + epsilon*x[1]**2)) + 1

    def f_prime(x):
        return -.5*np.array((
                    gaussian_prime(.5*x[0])*gaussian(.5*epsilon*x[1]),
                    gaussian(.5*x[0])*gaussian_prime(.5*epsilon*x[1]),
                    ))

    def hessian(x):
        return -.25*np.array((
                (gaussian_prime_prime(.5*x[0])*gaussian(.5*epsilon*x[1]),
                    gaussian_prime(.5*x[0])*gaussian_prime(.5*epsilon*x[1])),
                (gaussian_prime(.5*x[0])*gaussian_prime(.5*epsilon*x[1]),
                    gaussian(.5*x[0])*gaussian_prime_prime(.5*epsilon*x[1])),
                ))

    return f, f_prime, hessian


def mk_quad(epsilon):
    def f(x):
       return .33*(x[0]**2 + epsilon*x[1]**2)

    def f_prime(x):
       return .33*np.array((2*x[0], 2*epsilon*x[1]))

    def hessian(x):
       return .33*np.array([
                            [2, 0],
                            [0, 2*epsilon],
                           ])

    return f, f_prime, hessian

###############################################################################
# A gradient descent algorithm
# do not use: its a toy, use scipy's optimize.fmin_cg

def gradient_descent(x0, f, f_prime, hessian=None, adaptative=False):
    x_i, y_i = x0
    all_x_i = list()
    all_y_i = list()
    all_f_i = list()

    for i in range(1, 100):
        all_x_i.append(x_i)
        all_y_i.append(y_i)
        all_f_i.append(f([x_i, y_i]))
        dx_i, dy_i = f_prime([x_i, y_i])
        if adaptative:
            # Compute a step size using a line_search to satisfy the Wolf
            # conditions
            step = optimize.line_search(f, f_prime,
                                np.r_[x_i, y_i], -np.r_[dx_i, dy_i],
                                np.r_[dx_i, dy_i], c2=.05)
            step = step[0]
        else:
            step = 1
        x_i += - step*dx_i
        y_i += - step*dy_i
        if np.abs(all_f_i[-1]) < 1e-16:
            break
    return all_x_i, all_y_i, all_f_i


def gradient_descent_adaptative(x0, f, f_prime, hessian=None):
    return gradient_descent(x0, f, f_prime, adaptative=True)


def conjugate_gradient(x0, f, f_prime, hessian=None):
    all_x_i = [x0[0]]
    all_y_i = [x0[1]]
    all_f_i = [f(x0)]
    def store(X):
        x, y = X
        all_x_i.append(x)
        all_y_i.append(y)
        all_f_i.append(f(X))
    optimize.fmin_cg(f, x0, f_prime, callback=store, gtol=1e-12)
    return all_x_i, all_y_i, all_f_i


def newton_cg(x0, f, f_prime, hessian):
    all_x_i = [x0[0]]
    all_y_i = [x0[1]]
    all_f_i = [f(x0)]
    def store(X):
        x, y = X
        all_x_i.append(x)
        all_y_i.append(y)
        all_f_i.append(f(X))
    optimize.fmin_ncg(f, x0, f_prime, fhess=hessian, callback=store,
                avextol=1e-12)
    return all_x_i, all_y_i, all_f_i

def bfgs(x0, f, f_prime, hessian=None):
    all_x_i = [x0[0]]
    all_y_i = [x0[1]]
    all_f_i = [f(x0)]
    def store(X):
        x, y = X
        all_x_i.append(x)
        all_y_i.append(y)
        all_f_i.append(f(X))
    optimize.fmin_bfgs(f, x0, f_prime, callback=store, gtol=1e-12)
    return all_x_i, all_y_i, all_f_i



###############################################################################
# Run different optimizers on these problems

for index, ((f, f_prime, hessian), optimizer) in enumerate((
                (mk_quad(.7), gradient_descent),
                (mk_quad(.7), gradient_descent_adaptative),
                (mk_quad(.02), gradient_descent),
                (mk_quad(.02), gradient_descent_adaptative),
                (mk_gauss(.02), gradient_descent_adaptative),
                ((rosenbrock, rosenbrock_prime, rosenbrock_hessian),
                                    gradient_descent_adaptative),
                (mk_gauss(.02), conjugate_gradient),
                ((rosenbrock, rosenbrock_prime, rosenbrock_hessian),
                                    conjugate_gradient),
                (mk_quad(.02), newton_cg),
                (mk_gauss(.02), newton_cg),
                ((rosenbrock, rosenbrock_prime, rosenbrock_hessian),
                                    newton_cg),
                (mk_quad(.02), bfgs),
                (mk_gauss(.02), bfgs),
                ((rosenbrock, rosenbrock_prime, rosenbrock_hessian),
                            bfgs),
            )):
    x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    x = x.T
    y = y.T

    pl.figure(index, figsize=(3, 2.5))
    pl.clf()
    pl.axes([0, 0, 1, 1])

    X = np.concatenate((x[np.newaxis, ...], y[np.newaxis, ...]), axis=0)
    z = f([x, y])
    log_z = np.log(z + .01)
    pl.imshow(log_z,
            extent=[x_min, x_max, y_min, y_max],
            cmap=pl.cm.gray_r, origin='lower',
            vmax=log_z.min() + 1.5*log_z.ptp())
    contours = pl.contour(log_z,
                        extent=[x_min, x_max, y_min, y_max],
                        cmap=pl.cm.gnuplot, origin='lower')
    pl.clabel(contours, inline=1,
                fmt=super_fmt, fontsize=14)

    # Compute a gradient-descent
    x_i, y_i = 1.6, 1.1
    all_x_i, all_y_i, all_f_i = optimizer(np.array([x_i, y_i]),
                                    f, f_prime, hessian=hessian)

    pl.plot(all_x_i, all_y_i, 'b-', linewidth=2)
    pl.plot(all_x_i, all_y_i, 'k+')

    pl.plot([0], [0], 'rx', markersize=12)
    pl.xticks(())
    pl.yticks(())
    pl.draw()

    pl.figure(index + 100, figsize=(4, 3))
    pl.clf()
    pl.semilogy(np.abs(all_f_i), linewidth=2)
    pl.ylabel('Error on f(x)')
    pl.xlabel('Iteration')
    pl.tight_layout()
    pl.draw()

