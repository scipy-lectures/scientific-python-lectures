==========================
Mathematical optimization
==========================

:authors: Gaël Varoquaux

`Mathematical optimization
<http://en.wikipedia.org/wiki/Mathematical_optimization>`_ deals with the
problem of finding numerically minimums (or maximums or zeros) of
a function. In this context, the function is called *cost function*, or
*objective function*, or *energy*.

Here, we are interested in using :mod:`scipy.optimize` for black-box
optimization: we do not rely on the mathematical expression of the
function that we are optimizing. Note that this expression can often be
used for more efficient, non black-box, optimization.

.. topic:: Prerequisites

    * Numpy, Scipy
    * IPython
    * matplotlib

.. contents:: Chapters contents
   :local:
   :depth: 4

.. XXX: should I discuss root finding?

..
  For doctesting
  >>> import numpy as np

Knowning your problem
======================

Not all optimization problems are equal. Knowing your problem enables you
to choose the right tool.

.. topic:: **Dimensionality of the problem**

    The scale of an optimization problem is pretty much set by the
    *dimensionality of the problem*, i.e. the number of scalar variables
    on which the search is performed.

Convex versus non-convex optimization
---------------------------------------

.. |convex_1d_1| image:: auto_examples/images/plot_convex_1.png

.. |convex_1d_2| image:: auto_examples/images/plot_convex_2.png

.. list-table::

 * - |convex_1d_1|
 
   - |convex_1d_2|

 * - **A convex function**: 
 
     - `f` is above all its tangents.                    
     - equivalently, for two point A, B, f(C) lies below the segment 
       [f(A), f(B])], if A < C < B

   - **A non-convex function**

**Optimizing convex functions is easy. Optimizing non-convex functions can
be very hard.**

.. note:: A convex function provably has only one minimum, no local
   minimums

Smooth and non-smooth problems
-------------------------------

.. |smooth_1d_1| image:: auto_examples/images/plot_smooth_1.png

.. |smooth_1d_2| image:: auto_examples/images/plot_smooth_2.png

.. list-table::

 * - |smooth_1d_1|
 
   - |smooth_1d_2|

 * - **A smooth function**: 

     The gradient is defined everywhere, and is a continuous function
 
   - **A non-smooth function**

**Optimizing smooth functions is easier.**


Noisy versus exact cost functions
----------------------------------

.. |noisy| image:: auto_examples/images/plot_noisy_1.png

.. list-table::

 * - Noisy (blue) and non-noisy (green) functions
 
   - |noisy|

.. topic:: **Noisy gradients**

   Many optimization methods rely on gradients of the objective function.
   If the gradient function is not given, they are computed numerically,
   which induces errors. In such situation, even if the objective
   function is not noisy, 

Constraints
------------

.. |constraints| image:: auto_examples/images/plot_constraints_1.png

.. list-table::

 * - Optimizations under constraints

     Here: 
     
     :math:`-1 < x_1 < 1`
     
     :math:`-1 < x_2 < 1`
 
   - |constraints|


Getting started: 1D optimization
================================

The :func:`scipy.optimize.brent` can be used for efficient minimization of 1D functions.
It combines a bracketing strategy with a parabolic approximation.

.. |1d_optim_1| image:: auto_examples/images/plot_1d_optim_1.png
   :scale: 90%

.. |1d_optim_2| image:: auto_examples/images/plot_1d_optim_2.png
   :scale: 75%

.. |1d_optim_3| image:: auto_examples/images/plot_1d_optim_3.png
   :scale: 90%

.. |1d_optim_4| image:: auto_examples/images/plot_1d_optim_4.png
   :scale: 75%

.. list-table::

 * - **Brent's method on a quadratic function**: it converges in 3 iterations,
     as the quadratic approximation is then exact.

   - |1d_optim_1|
 
   - |1d_optim_2|

 * - **Brent's method on a non-convex function**: note that the fact that the
     optimizer avoided the local minimum is a matter of luck.

   - |1d_optim_3|

   - |1d_optim_4|

::

    >>> from scipy import optimize
    >>> def f(x):
    ...     return -np.exp(-(x - .7)**2)
    >>> x_min = optimize.brent(f)  # It actually converges in 9 iterations!
    >>> x_min #doctest: +ELLIPSIS
    0.6999999997759...
    >>> x_min - .7 #doctest: +ELLIPSIS
    -2.1605...e-10

.. note:: 
   
   Brent's method can be used for optimization constraint to an
   intervale using :func:`scipy.optimize.fminbound`

.. note::
   
   In scipy 0.11, :func:`scipy.optimize.minimize_scalar` gives a generic
   interface to 1D scalar minimization

Gradient based methods
========================================

Some intuitions about gradient descent
---------------------------------------

Here we focus on **intuitions**, not code. Code will follow.

`Gradient descent <http://en.wikipedia.org/wiki/Gradient_descent>`_
basically consists consists in taking small steps in the direction of the
gradient.

.. |gradient_quad_cond| image:: auto_examples/images/plot_gradient_descent_0.png
   :scale: 90%

.. |gradient_quad_cond_conv| image:: auto_examples/images/plot_gradient_descent_100.png
   :scale: 75%

.. |gradient_quad_icond| image:: auto_examples/images/plot_gradient_descent_2.png
   :scale: 90%

.. |gradient_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_102.png
   :scale: 75%

.. list-table:: **Fixed step gradient descent**

 * - **A well-conditionned quadratic function.**

   - |gradient_quad_cond|
 
   - |gradient_quad_cond_conv|

 * - **An ill-conditionned quadratic function.**

     The core problem of gradient-methods on ill-conditionned problems is
     that the gradient tends not to point in the direction of the
     minimum.

   - |gradient_quad_icond|
 
   - |gradient_quad_icond_conv|

We can see that very anisotropic (`ill-conditionned
<http://en.wikipedia.org/wiki/Condition_number>`_) functions are harder
to optimize.

.. topic:: **Take home message: preconditionning**

   If you know natural scaling for your variables, prescale them so that
   they behave similarly. This is related to `preconditionning
   <http://en.wikipedia.org/wiki/Preconditioner>`_.

Also, it clearly can clearly be advantageous to take bigger steps. This
is done in gradient descent code using a
`line search <http://en.wikipedia.org/wiki/Line_search>`_.

.. |agradient_quad_cond| image:: auto_examples/images/plot_gradient_descent_1.png
   :scale: 90%

.. |agradient_quad_cond_conv| image:: auto_examples/images/plot_gradient_descent_101.png
   :scale: 75%

.. |agradient_quad_icond| image:: auto_examples/images/plot_gradient_descent_3.png
   :scale: 90%

.. |agradient_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_103.png
   :scale: 75%

.. |agradient_gauss_icond| image:: auto_examples/images/plot_gradient_descent_4.png
   :scale: 90%

.. |agradient_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_104.png
   :scale: 75%

.. |agradient_rosen_icond| image:: auto_examples/images/plot_gradient_descent_5.png
   :scale: 90%

.. |agradient_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_105.png
   :scale: 75%


.. list-table:: **Adaptive step gradient descent**

 * - A well-conditionned quadratic function.

   - |agradient_quad_cond|
 
   - |agradient_quad_cond_conv|

 * - An ill-conditionned quadratic function.

   - |agradient_quad_icond|
 
   - |agradient_quad_icond_conv|

 * - An ill-conditionned non-quadratic function.

   - |agradient_gauss_icond|
 
   - |agradient_gauss_icond_conv|

 * - An ill-conditionned very non-quadratic function.

   - |agradient_rosen_icond|
 
   - |agradient_rosen_icond_conv|

The more a function looks like a quadratic function (elliptic
iso-curves), the easier it is to optimize.

Conjugate gradient descent
---------------------------

The gradient descent algorithms above are toys not to be used on real
problems.

As can be seen from the above experiments, one of the problems of the
simple gradient descent algorithms, is that it tends to oscillate across
a valley, each time following the direction of the gradient, that makes
it cross the valley. The conjugate gradient solves this problem by adding
a *friction* term: each step depends on the two last values of the
gradient and sharp turns are reduced.

.. |cg_gauss_icond| image:: auto_examples/images/plot_gradient_descent_6.png
   :scale: 90%

.. |cg_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_106.png
   :scale: 75%

.. |cg_rosen_icond| image:: auto_examples/images/plot_gradient_descent_7.png
   :scale: 90%

.. |cg_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_107.png
   :scale: 75%


.. list-table:: **Conjugate gradient descent**

 * - An ill-conditionned non-quadratic function.

   - |cg_gauss_icond|
 
   - |cg_gauss_icond_conv|

 * - An ill-conditionned very non-quadratic function.

   - |cg_rosen_icond|
 
   - |cg_rosen_icond_conv|

Methods based on conjugate gradient are named with *'cg'* in scipy. The
simple conjugate gradient method to minimize a function is
:func:`scipy.optimize.fmin_cg`::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> optimize.fmin_cg(f, [2, 2])
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 13
            Function evaluations: 120
            Gradient evaluations: 30
    array([ 0.99998968,  0.99997855])

These methods need the gradient of the function. They can compute it, but
will perform better if you can pass them the gradient::

    >>> def fprime(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.fmin_cg(f, [2, 2], fprime=fprime)
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 13
            Function evaluations: 30
            Gradient evaluations: 30
    array([ 0.99999199,  0.99997536])

Note that the function has only been evaluated 30 times, compared to 120
without the gradient.

Computing gradients
--------------------

XXX: TODO

.. warning::
   
   A common source of optimization not converging is human error in the
   computation of the gradient. You can use
   :func:`scipy.optimize.check_grad` to check that your gradient is
   correct. It returns the norm of the different between the gradient
   given, and a gradient computed numerically:

    >>> optimize.check_grad(f, fprime, [2, 2])
    2.384185791015625e-07


Newton and quasi-newton methods
================================

Newton methods: using the Hessian (2nd differential)
------------------------------------------------------

`Newton methods
<http://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>`_ use a
local quadratic approximation to compute the jump direction. For this
purpose, they rely on the 2 first derivative of the function: the
*gradient* and the `Hessian
<http://en.wikipedia.org/wiki/Hessian_matrix>`_.

.. |ncg_quad_icond| image:: auto_examples/images/plot_gradient_descent_8.png
   :scale: 90%

.. |ncg_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_108.png
   :scale: 75%

.. |ncg_gauss_icond| image:: auto_examples/images/plot_gradient_descent_9.png
   :scale: 90%

.. |ncg_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_109.png
   :scale: 75%

.. |ncg_rosen_icond| image:: auto_examples/images/plot_gradient_descent_10.png
   :scale: 90%

.. |ncg_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_110.png
   :scale: 75%


.. list-table::

 * - **An ill-conditionned quadratic function:**

     Note that, as the quadratic approximation is exact, the Newton
     method is blazing fast

   - |ncg_quad_icond|
 
   - |ncg_quad_icond_conv|

 * - **An ill-conditionned non-quadratic function:**

     Here we are optimizing a Gaussian, which is always below its
     quadratic approximation. As a result, the Newton method overshoots
     and leads to oscillations.

   - |ncg_gauss_icond|
 
   - |ncg_gauss_icond_conv|

 * - **An ill-conditionned very non-quadratic function:**

   - |ncg_rosen_icond|
 
   - |ncg_rosen_icond_conv|

In scipy, the Newton method for optimization is implemented in
:func:`scipy.optimize.fmin_ncg` (cg here refers to that fact that an
inner operation, the inversion of the Hessian, is performed by conjugate
gradient). :func:`scipy.optimize.fmin_tnc` can be use for constraint
problems, although it is less versatile::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def fprime(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.fmin_ncg(f, [2, 2], fprime=fprime)
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 10
            Function evaluations: 12
            Gradient evaluations: 44
            Hessian evaluations: 0
    array([ 1.,  1.])

Note that compared to a conjugate gradient (above), Newton's method has
required less function evaluations, but more gradient evaluations, as it
uses it to approximate the Hessian. Let's compute the Hessian and pass it
to the algorithm::

    >>> def hessian(x): # Computed with sympy
    ...     return np.array(((1 - 4*x[1] + 12*x[0]**2, -4*x[0]), (-4*x[0], 2)))
    >>> optimize.fmin_ncg(f, [2, 2], fprime=fprime, fhess=hessian)
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 10
            Function evaluations: 12
            Gradient evaluations: 10
            Hessian evaluations: 10
    array([ 1.,  1.])



XXX: remark on the fact that at high-dimension, the inversion of the
Hessian is costly and unstable (large scale = 250).

.. note:: 
   
    Newton optimizers should not to be confused with Newton's root finding
    method, based on the same principles, :func:`scipy.optimize.newton`.

Quasi-Newton methods: approximating the Hessian on the fly 
------------------------------------------------------------

**BFGS**: BFGS (Broyden-Fletcher-Goldfarb-Shanno algorithm) refines at
each step an approximation of the Hessian.

.. |bfgs_quad_icond| image:: auto_examples/images/plot_gradient_descent_11.png
   :scale: 90%

.. |bfgs_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_111.png
   :scale: 75%

.. |bfgs_gauss_icond| image:: auto_examples/images/plot_gradient_descent_12.png
   :scale: 90%

.. |bfgs_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_112.png
   :scale: 75%

.. |bfgs_rosen_icond| image:: auto_examples/images/plot_gradient_descent_13.png
   :scale: 90%

.. |bfgs_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_113.png
   :scale: 75%


.. list-table::

 * - **An ill-conditionned quadratic function:**

     On a exactly quadratic function, BFGS is not as fast as Newton's
     method, but still very fast.

   - |bfgs_quad_icond|
 
   - |bfgs_quad_icond_conv|

 * - **An ill-conditionned non-quadratic function:**

     Here BFGS does better than Newton, as its empirical estimate of the
     curvature is better than that given by the Hessian.

   - |bfgs_gauss_icond|
 
   - |bfgs_gauss_icond_conv|

 * - **An ill-conditionned very non-quadratic function:**

   - |bfgs_rosen_icond|
 
   - |bfgs_rosen_icond_conv|

::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def fprime(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.fmin_bfgs(f, [2, 2], fprime=fprime)
    Optimization terminated successfully.
            Current function value: 0.000000
            Iterations: 16
            Function evaluations: 24
            Gradient evaluations: 24
    array([ 1.00000017,  1.00000026])


**L-BFGS:** Limited-memory BFGS Sits between BFGS and conjugate gradient:
in very high dimensions (> 250) the Hessian matrix is too costly to
compute and invert. L-BFGS keeps a low-rank version. In addition, the
scipy version, :func:`scipy.optimize.fmin_l_bfgs_b`, includes box bounds::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def fprime(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.fmin_l_bfgs_b(f, [2, 2], fprime=fprime)
    (array([ 1.00000005,  1.00000009]), 1.4417677473011859e-15, {'warnflag': 0, 'task': 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL', 'grad': array([  1.02331202e-07,  -2.59299369e-08]), 'funcalls': 17})

Gradient-less methods
======================

A shooting method: the Powell algorithm
----------------------------------------

Almost a gradient approach

.. |powell_quad_icond| image:: auto_examples/images/plot_gradient_descent_14.png
   :scale: 90%

.. |powell_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_114.png
   :scale: 75%

.. |powell_gauss_icond| image:: auto_examples/images/plot_gradient_descent_15.png
   :scale: 90%

.. |powell_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_115.png
   :scale: 75%


.. |powell_rosen_icond| image:: auto_examples/images/plot_gradient_descent_16.png
   :scale: 90%

.. |powell_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_116.png
   :scale: 75%


.. list-table::

 * - **An ill-conditionned quadratic function:**

     Powell's method isn't too sensitive to local ill-conditionning in
     low dimensions

   - |powell_quad_icond|
 
   - |powell_quad_icond_conv|

 * - **An ill-conditionned non-quadratic function:**

   - |powell_gauss_icond|
 
   - |powell_gauss_icond_conv|

 * - **An ill-conditionned very non-quadratic function:**

   - |powell_rosen_icond|
 
   - |powell_rosen_icond_conv|


Simplex method: the Nelder-Mead
--------------------------------

Nelder-Mead: robust, but slower on smooth, non-noisy functions

.. |nm_gauss_icond| image:: auto_examples/images/plot_gradient_descent_17.png
   :scale: 90%

.. |nm_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_117.png
   :scale: 75%


.. |nm_rosen_icond| image:: auto_examples/images/plot_gradient_descent_18.png
   :scale: 90%

.. |nm_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_118.png
   :scale: 75%


.. list-table::

 * - **An ill-conditionned non-quadratic function:**

   - |nm_gauss_icond|
 
   - |nm_gauss_icond_conv|

 * - **An ill-conditionned very non-quadratic function:**

   - |nm_rosen_icond|
 
   - |nm_rosen_icond_conv|


Global optimizers
==================

Comparion of generic methods
=============================

* Newton requires the Hessian of the problem.

* On very ill-conditioned problems BFGS, is equivalent to gradient
  descent. Use `preconditionning
  <http://en.wikipedia.org/wiki/Preconditioner>`_ when possible.
  Conjugate gradient is independent of the conditioning (though it
  converges slower on ill-conditioned problems).

* Computational overhead of BFGS is larger than that of conjugate
  gradient. On the other side, one iteration of BFGS usually needs less
  function evaluations than CG (up to 2 times less). Thus conjugate
  gradient method is better than BFGS at optimizing computationally cheap
  functions.

Special case: least-squares
============================

linalg.leastsq

optimize.curve_fit


Optimization with constraints
==============================

SLSQP
Cobyla
fmin_bound
L-BFGS-B

Alternate optimization: block coordinate methods
=================================================


