"""
Example cost functions or objective functions to optimize.
"""
import numpy as np

###############################################################################
# Gaussian functions with varying conditionning

def gaussian(x):
    return np.exp(-np.sum(x**2))


def gaussian_prime(x):
    return -2*x*np.exp(-np.sum(x**2))


def gaussian_prime_prime(x):
    return -2*np.exp(-x**2) + 4*x**2*np.exp(-x**2)


def mk_gauss(epsilon, ndim=2):
    def f(x):
        x = np.asarray(x)
        y = x.copy()
        y *= np.power(epsilon, np.arange(ndim))
        return -gaussian(.5*y) + 1

    def f_prime(x):
        x = np.asarray(x)
        y = x.copy()
        scaling = np.power(epsilon, np.arange(ndim))
        y *= scaling
        return -.5*scaling*gaussian_prime(.5*y)

    def hessian(x):
        epsilon = .07
        x = np.asarray(x)
        y = x.copy()
        scaling = np.power(epsilon, np.arange(ndim))
        y *= .5*scaling
        H = -.25*np.ones((ndim, ndim))*gaussian(y)
        d = 4*y*y[:, np.newaxis]
        d.flat[::ndim+1] += -2
        H *= d
        return H

    return f, f_prime, hessian

###############################################################################
# Quadratic functions with varying conditionning

def mk_quad(epsilon, ndim=2):
    def f(x):
       x = np.asarray(x)
       y = x.copy()
       y *= np.power(epsilon, np.arange(ndim))
       return .33*np.sum(y**2)

    def f_prime(x):
       x = np.asarray(x)
       y = x.copy()
       scaling = np.power(epsilon, np.arange(ndim))
       y *= scaling
       return .33*2*scaling*y

    def hessian(x):
       scaling = np.power(epsilon, np.arange(ndim))
       return .33*2*np.diag(scaling)

    return f, f_prime, hessian


###############################################################################
# Super ill-conditionned problem: the Rosenbrock function

def rosenbrock(x):
    y = 4*x
    y[0] += 1
    y[1:] += 3
    return np.sum(.5*(1 - y[:-1])**2 + (y[1:] - y[:-1]**2)**2)


def rosenbrock_prime(x):
    y = 4*x
    y[0] += 1
    y[1:] += 3
    xm = y[1:-1]
    xm_m1 = y[:-2]
    xm_p1 = y[2:]
    der = np.zeros_like(y)
    der[1:-1] = 2*(xm - xm_m1**2) - 4*(xm_p1 - xm**2)*xm - .5*2*(1 - xm)
    der[0] = -4*y[0]*(y[1] - y[0]**2) - .5*2*(1 - y[0])
    der[-1] = 2*(y[-1] - y[-2]**2)
    return 4*der


def rosenbrock_hessian_(x):
    x, y = x
    x = 4*x + 1
    y = 4*y + 3
    return 4*4*np.array((
                    (1 - 4*y + 12*x**2, -4*x),
                    (             -4*x,    2),
                   ))


def rosenbrock_hessian(x):
    y = 4*x
    y[0] += 1
    y[1:] += 3

    H = np.diag(-4*y[:-1], 1) - np.diag(4*y[:-1], -1)
    diagonal = np.zeros_like(y)
    diagonal[0] = 12*y[0]**2 - 4*y[1] + 2*.5
    diagonal[-1] = 2
    diagonal[1:-1] = 3 + 12*y[1:-1]**2 - 4*y[2:]*.5
    H = H + np.diag(diagonal)
    return 4*4*H


###############################################################################
# Helpers to wrap the functions

class LoggingFunction(object):

    def __init__(self, function, counter=None):
        self.function = function
        if counter is None:
            counter = list()
        self.counter = counter
        self.all_x_i = list()
        self.all_y_i = list()
        self.all_f_i = list()
        self.counts = list()

    def __call__(self, x0):
        x_i, y_i = x0[:2]
        self.all_x_i.append(x_i)
        self.all_y_i.append(y_i)
        f_i = self.function(np.asarray(x0))
        self.all_f_i.append(f_i)
        self.counter.append('f')
        self.counts.append(len(self.counter))
        return f_i

class CountingFunction(object):

    def __init__(self, function, counter=None):
        self.function = function
        if counter is None:
            counter = list()
        self.counter = counter

    def __call__(self, x0):
        self.counter.append('f_prime')
        return self.function(x0)



