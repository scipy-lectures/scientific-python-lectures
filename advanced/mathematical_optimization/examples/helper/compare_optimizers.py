"""
Comparing optimizers
=====================

Comparison of optimizers on various problems.
"""

import functools
import pickle
import sys

import numpy as np
import scipy as sp
from joblib import Memory

from cost_functions import (
    mk_quad,
    mk_gauss,
    rosenbrock,
    rosenbrock_prime,
    rosenbrock_hessian,
    LoggingFunction,
    CountingFunction,
)


def my_partial(**kwargs):
    function = sp.optimize.minimize
    f = functools.partial(function, **kwargs)
    functools.update_wrapper(f, function)
    return f


methods = {
    "Nelder-mead": my_partial(
        method="Nelder-Mead",
        options={"fatol": 1e-12, "maxiter": 5e3, "xatol": 1e-7, "maxfev": 1e6},
    ),
    "Powell": my_partial(
        method="Powell", options={"ftol": 1e-9, "maxiter": 5e3, "maxfev": 1e7}
    ),
    "BFGS": my_partial(method="BFGS", options={"gtol": 1e-9, "maxiter": 5e3}),
    "Newton": my_partial(method="Newton-CG", options={"xtol": 1e-7, "maxiter": 5e3}),
    "Conjugate gradient": my_partial(
        method="CG", options={"gtol": 1e-7, "maxiter": 5e3}
    ),
    "L-BFGS": my_partial(
        method="L-BFGS-B", options={"ftol": 10.0, "gtol": 1e-8, "maxfun": 1e7}
    ),
    "L-BFGS w f'": my_partial(
        method="L-BFGS-B", options={"ftol": 10.0, "gtol": 1e-8, "maxfun": 1e7}
    ),
}

###############################################################################


def bencher(cost_name, ndim, method_name, x0):
    cost_function = mk_costs(ndim)[0][cost_name][0]
    method = methods[method_name]
    f = LoggingFunction(cost_function)
    method(f, x0)
    this_costs = np.array(f.all_f_i)
    return this_costs


# Bench with gradients
def bencher_gradient(cost_name, ndim, method_name, x0):
    cost_function, cost_function_prime, hessian = mk_costs(ndim)[0][cost_name]
    method = methods[method_name]
    f_prime = CountingFunction(cost_function_prime)
    f = LoggingFunction(cost_function, counter=f_prime.counter)
    method(f, x0, jac=f_prime)
    this_costs = np.array(f.all_f_i)
    return this_costs, np.array(f.counts)


# Bench with the hessian
def bencher_hessian(cost_name, ndim, method_name, x0):
    cost_function, cost_function_prime, hessian = mk_costs(ndim)[0][cost_name]
    method = methods[method_name]
    f_prime = CountingFunction(cost_function_prime)
    hessian = CountingFunction(hessian, counter=f_prime.counter)
    f = LoggingFunction(cost_function, counter=f_prime.counter)
    method(f, x0, jac=f_prime, hess=hessian)
    this_costs = np.array(f.all_f_i)
    return this_costs, np.array(f.counts)


def mk_costs(ndim=2):
    costs = {
        "Well-conditioned quadratic": mk_quad(0.7, ndim=ndim),
        "Ill-conditioned quadratic": mk_quad(0.02, ndim=ndim),
        "Well-conditioned Gaussian": mk_gauss(0.7, ndim=ndim),
        "Ill-conditioned Gaussian": mk_gauss(0.02, ndim=ndim),
        "Rosenbrock  ": (rosenbrock, rosenbrock_prime, rosenbrock_hessian),
    }

    rng = np.random.default_rng(5982345892)
    starting_points = 4 * rng.random((20, ndim)) - 2
    if ndim > 100:
        starting_points = starting_points[:10]
    return costs, starting_points


###############################################################################
# Compare methods without gradient
mem = Memory(".", verbose=3)

if True:
    gradient_less_benchs = {}

    for ndim in (2, 8, 32, 128):
        this_dim_benchs = {}
        costs, starting_points = mk_costs(ndim)
        for cost_name, cost_function in costs.items():
            # We don't need the derivative or the hessian
            cost_function = cost_function[0]
            function_bench = {}
            for x0 in starting_points:
                all_bench = []
                # Bench gradient-less
                for method_name, method in methods.items():
                    if method_name in ("Newton", "L-BFGS w f'"):
                        continue
                    this_bench = function_bench.get(method_name, [])
                    this_costs = mem.cache(bencher)(cost_name, ndim, method_name, x0)
                    if np.all(this_costs > 0.25 * ndim**2 * 1e-9):
                        convergence = 2 * len(this_costs)
                    else:
                        convergence = (
                            np.where(np.diff(this_costs > 0.25 * ndim**2 * 1e-9))[
                                0
                            ].max()
                            + 1
                        )
                    this_bench.append(convergence)
                    all_bench.append(convergence)
                    function_bench[method_name] = this_bench

                # Bench with gradients
                for method_name, method in methods.items():
                    if method_name in ("Newton", "Powell", "Nelder-mead", "L-BFGS"):
                        continue
                    this_method_name = method_name
                    if method_name.endswith(" w f'"):
                        this_method_name = method_name[:-4]
                    this_method_name = this_method_name + "\nw f'"
                    this_bench = function_bench.get(this_method_name, [])
                    this_costs, this_counts = mem.cache(bencher_gradient)(
                        cost_name, ndim, method_name, x0
                    )
                    if np.all(this_costs > 0.25 * ndim**2 * 1e-9):
                        convergence = 2 * this_counts.max()
                    else:
                        convergence = (
                            np.where(np.diff(this_costs > 0.25 * ndim**2 * 1e-9))[
                                0
                            ].max()
                            + 1
                        )
                        convergence = this_counts[convergence]
                    this_bench.append(convergence)
                    all_bench.append(convergence)
                    function_bench[this_method_name] = this_bench

                # Bench Newton with Hessian
                method_name = "Newton"
                this_bench = function_bench.get(method_name, [])
                this_costs, this_counts = mem.cache(bencher_hessian)(
                    cost_name, ndim, method_name, x0
                )
                if np.all(this_costs > 0.25 * ndim**2 * 1e-9):
                    convergence = 2 * len(this_costs)
                else:
                    convergence = (
                        np.where(np.diff(this_costs > 0.25 * ndim**2 * 1e-9))[0].max()
                        + 1
                    )
                this_bench.append(convergence)
                all_bench.append(convergence)
                function_bench[method_name + "\nw Hessian "] = this_bench

                # Normalize across methods
                x0_mean = np.mean(all_bench)
                for _, values in function_bench.items():
                    values[-1] /= x0_mean
            this_dim_benchs[cost_name] = function_bench
        gradient_less_benchs[ndim] = this_dim_benchs
        print(80 * "_")
        print(f"Done cost {cost_name}, ndim {ndim}")
        print(80 * "_")

    pickle.dump(
        gradient_less_benchs,
        open(f"compare_optimizers_py{sys.version_info[0]}.pkl", "wb"),
    )
