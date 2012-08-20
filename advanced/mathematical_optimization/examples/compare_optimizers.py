"""
Comparison of optimizers on various problems.
"""
import functools
import pickle

import numpy as np
from scipy import optimize
from joblib import Memory

from cost_functions import mk_quad, mk_gauss, rosenbrock,\
    rosenbrock_prime, rosenbrock_hessian, LoggingFunction

def my_partial(function, **kwargs):
    f = functools.partial(function, **kwargs)
    functools.update_wrapper(f, function)
    return f

methods = {
    'Nelder-mead':          my_partial(optimize.fmin,
                                        ftol=1e-12, maxiter=1e4,
                                        xtol=1e-7, maxfun=1e6),
    'Powell':               my_partial(optimize.fmin_powell,
                                        ftol=1e-10, maxiter=1e4,
                                        maxfun=1e7),
    'BFGS':                 my_partial(optimize.fmin_bfgs,
                                        gtol=1e-9, maxiter=1e4),
    'Newton':               my_partial(optimize.fmin_ncg,
                                        avextol=1e-10, maxiter=1e4),
    'Conjugate gradient':   my_partial(optimize.fmin_cg,
                                        gtol=1e-9, maxiter=1e4),
    'L-BFGS':               my_partial(optimize.fmin_l_bfgs_b,
                                        approx_grad=1, factr=10.0,
                                        pgtol=1e-8, maxfun=1e7),
}

def bencher(cost_name, ndim, method_name, x0):
    cost_function = mk_costs(ndim)[0][cost_name][0]
    method = methods[method_name]
    f = LoggingFunction(cost_function)
    method(f, x0)
    this_costs = np.array(f.all_f_i)
    return this_costs

# XXX: should vary the dimensionality

def mk_costs(ndim=2):
    costs = {
            'Well-conditionned quadratic':      mk_quad(.7, ndim=ndim),
            'Ill-conditionned quadratic':       mk_quad(.02, ndim=ndim),
            'Well-conditionned Gaussian':       mk_gauss(.7, ndim=ndim),
            'Ill-conditionned Gaussian':        mk_gauss(.02, ndim=ndim),
            'Rosenbrock':   (rosenbrock, rosenbrock_prime, rosenbrock_hessian),
        }

    rng = np.random.RandomState(0)
    starting_points = 4*rng.rand(20, ndim) - 2
    return costs, starting_points

###############################################################################
# Compare methods without gradient
mem = Memory('.', verbose=3)

gradient_less_benchs = dict()

for ndim in (2, 8, 32):
    this_dim_benchs = dict()
    costs, starting_points = mk_costs(ndim)
    for cost_name, cost_function in costs.iteritems():
        # We don't need the derivative or the hessian
        cost_function = cost_function[0]
        function_bench = dict()
        for x0 in starting_points:
            all_bench = list()
            for method_name, method in methods.iteritems():
                if method_name == 'Newton':
                    continue
                this_bench = function_bench.get(method_name, list())
                this_costs = mem.cache(bencher)(cost_name, ndim,
                                                method_name, x0)
                if np.all(this_costs > .25*ndim**2*1e-9):
                    convergence = len(this_costs)
                else:
                    convergence = np.where(
                                        np.diff(this_costs > .25*ndim**2*1e-9)
                                    )[0].max() + 1
                this_bench.append(convergence)
                all_bench.append(convergence)
                function_bench[method_name] = this_bench
            # Normalize across methods
            x0_mean = np.mean(all_bench)
            for method_name in function_bench:
                function_bench[method_name][-1] /= x0_mean
        this_dim_benchs[cost_name] = function_bench
    gradient_less_benchs[ndim] = this_dim_benchs
    print 80*'_'
    print 'Done cost %s, ndim %s' % (cost_name, ndim)
    print 80*'_'


pickle.dump(gradient_less_benchs, file('compare_optimizers.pkl', 'w'))
