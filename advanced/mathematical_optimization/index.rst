.. _mathematical_optimization:

=======================================================
Mathematical optimization: finding minima of functions
=======================================================

**Authors**: *Gaël Varoquaux*

`Mathematical optimization
<https://en.wikipedia.org/wiki/Mathematical_optimization>`_ deals with the
problem of finding numerically minimums (or maximums or zeros) of
a function. In this context, the function is called *cost function*, or
*objective function*, or *energy*.

Here, we are interested in using :mod:`scipy.optimize` for black-box
optimization: we do not rely on the mathematical expression of the
function that we are optimizing. Note that this expression can often be
used for more efficient, non black-box, optimization.

.. topic:: Prerequisites

    * Numpy, Scipy
    * matplotlib

.. seealso::  **References**

   Mathematical optimization is very ... mathematical. If you want
   performance, it really pays to read the books:

   * `Convex Optimization <http://www.stanford.edu/~boyd/cvxbook/>`_ 
     by Boyd and Vandenberghe (pdf available free online).

   * `Numerical Optimization
     <http://users.eecs.northwestern.edu/~nocedal/book/num-opt.html>`_, 
     by Nocedal and Wright. Detailed reference on gradient descent methods.

   * `Practical Methods of Optimization
     <http://www.amazon.com/gp/product/0471494631/ref=ox_sc_act_title_1?ie=UTF8&smid=ATVPDKIKX0DER>`_ by Fletcher: good at hand-waving explanations.

.. include:: ../../includes/big_toc_css.rst


.. contents:: Chapters contents
   :local:
   :depth: 2

.. XXX: should I discuss root finding?

..
  For doctesting
  >>> import numpy as np

Knowing your problem
======================

Not all optimization problems are equal. Knowing your problem enables you
to choose the right tool.

.. topic:: **Dimensionality of the problem**

    The scale of an optimization problem is pretty much set by the
    *dimensionality of the problem*, i.e. the number of scalar variables
    on which the search is performed.

Convex versus non-convex optimization
---------------------------------------

.. |convex_1d_1| image:: auto_examples/images/sphx_glr_plot_convex_001.png

.. |convex_1d_2| image:: auto_examples/images/sphx_glr_plot_convex_002.png

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

.. note:: It can be proven that for a convex function a local minimum is
   also a global minimum. Then, in some sense, the minimum is unique.

Smooth and non-smooth problems
-------------------------------

.. |smooth_1d_1| image:: auto_examples/images/sphx_glr_plot_smooth_001.png

.. |smooth_1d_2| image:: auto_examples/images/sphx_glr_plot_smooth_002.png

.. list-table::

 * - |smooth_1d_1|
 
   - |smooth_1d_2|

 * - **A smooth function**: 

     The gradient is defined everywhere, and is a continuous function
 
   - **A non-smooth function**

**Optimizing smooth functions is easier**
(true in the context of *black-box* optimization, otherwise
`Linear Programming <http://en.wikipedia.org/wiki/Linear_programming>`_
is an example of methods which deal very efficiently with
piece-wise linear functions).



Noisy versus exact cost functions
----------------------------------

.. |noisy| image:: auto_examples/images/sphx_glr_plot_noisy_001.png

.. list-table::

 * - Noisy (blue) and non-noisy (green) functions
 
   - |noisy|

.. topic:: **Noisy gradients**

   Many optimization methods rely on gradients of the objective function.
   If the gradient function is not given, they are computed numerically,
   which induces errors. In such situation, even if the objective
   function is not noisy, a gradient-based optimization may be a noisy
   optimization.

Constraints
------------

.. |constraints| image:: auto_examples/images/sphx_glr_plot_constraints_001.png
    :target: auto_examples/plot_constraints.html

.. list-table::

 * - Optimizations under constraints

     Here: 
     
     :math:`-1 < x_1 < 1`
     
     :math:`-1 < x_2 < 1`
 
   - |constraints|


A review of the different optimizers
======================================

Getting started: 1D optimization
---------------------------------

Let's get started by finding the minimum of the scalar function
:math:`f(x)=\exp[(x-0.7)^2]`. :func:`scipy.optimize.minimize_scalar` uses
Brent's method to find the minimum of a function:

::

    >>> from scipy import optimize
    >>> def f(x):
    ...     return -np.exp(-(x - 0.7)**2)
    >>> result = optimize.minimize_scalar(f)
    >>> result.success # check if solver was successful
    True
    >>> x_min = result.x
    >>> x_min #doctest: +ELLIPSIS
    0.699999999...
    >>> x_min - 0.7 #doctest: +ELLIPSIS
    -2.160590595323697e-10


.. |1d_optim_1| image:: auto_examples/images/sphx_glr_plot_1d_optim_001.png
   :scale: 90%

.. |1d_optim_2| image:: auto_examples/images/sphx_glr_plot_1d_optim_002.png
   :scale: 75%

.. |1d_optim_3| image:: auto_examples/images/sphx_glr_plot_1d_optim_003.png
   :scale: 90%

.. |1d_optim_4| image:: auto_examples/images/sphx_glr_plot_1d_optim_004.png
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


.. note:: 
   
   You can use different solvers using the parameter ``method``.

.. note::

    :func:`scipy.optimize.minimize_scalar` can also be used for optimization
    constrained to an interval using the parameter ``bounds``.

Gradient based methods
-----------------------

Some intuitions about gradient descent
.......................................

Here we focus on **intuitions**, not code. Code will follow.

`Gradient descent <http://en.wikipedia.org/wiki/Gradient_descent>`_
basically consists in taking small steps in the direction of the
gradient, that is the direction of the *steepest descent*.

.. |gradient_quad_cond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_000.png
   :scale: 90%

.. |gradient_quad_cond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_100.png
   :scale: 75%

.. |gradient_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_002.png
   :scale: 90%

.. |gradient_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_102.png
   :scale: 75%

.. list-table:: **Fixed step gradient descent**

 * - **A well-conditioned quadratic function.**

   - |gradient_quad_cond|
 
   - |gradient_quad_cond_conv|

 * - **An ill-conditioned quadratic function.**

     The core problem of gradient-methods on ill-conditioned problems is
     that the gradient tends not to point in the direction of the
     minimum.

   - |gradient_quad_icond|
 
   - |gradient_quad_icond_conv|

We can see that very anisotropic (`ill-conditioned
<http://en.wikipedia.org/wiki/Condition_number>`_) functions are harder
to optimize.

.. topic:: **Take home message: conditioning number and preconditioning**

   If you know natural scaling for your variables, prescale them so that
   they behave similarly. This is related to `preconditioning
   <https://en.wikipedia.org/wiki/Preconditioner>`_.

Also, it clearly can be advantageous to take bigger steps. This
is done in gradient descent code using a
`line search <https://en.wikipedia.org/wiki/Line_search>`_.

.. |agradient_quad_cond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_001.png
   :scale: 90%

.. |agradient_quad_cond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_101.png
   :scale: 75%

.. |agradient_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_003.png
   :scale: 90%

.. |agradient_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_103.png
   :scale: 75%

.. |agradient_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_004.png
   :scale: 90%

.. |agradient_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_104.png
   :scale: 75%

.. |agradient_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_005.png
   :scale: 90%

.. |agradient_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_105.png
   :scale: 75%


.. list-table:: **Adaptive step gradient descent**

 * - A well-conditioned quadratic function.

   - |agradient_quad_cond|
 
   - |agradient_quad_cond_conv|

 * - An ill-conditioned quadratic function.

   - |agradient_quad_icond|
 
   - |agradient_quad_icond_conv|

 * - An ill-conditioned non-quadratic function.

   - |agradient_gauss_icond|
 
   - |agradient_gauss_icond_conv|

 * - An ill-conditioned very non-quadratic function.

   - |agradient_rosen_icond|
 
   - |agradient_rosen_icond_conv|

The more a function looks like a quadratic function (elliptic
iso-curves), the easier it is to optimize.

Conjugate gradient descent
...........................

The gradient descent algorithms above are toys not to be used on real
problems.

As can be seen from the above experiments, one of the problems of the
simple gradient descent algorithms, is that it tends to oscillate across
a valley, each time following the direction of the gradient, that makes
it cross the valley. The conjugate gradient solves this problem by adding
a *friction* term: each step depends on the two last values of the
gradient and sharp turns are reduced.

.. |cg_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_006.png
   :scale: 90%

.. |cg_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_106.png
   :scale: 75%

.. |cg_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_007.png
   :scale: 90%

.. |cg_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_107.png
   :scale: 75%


.. list-table:: **Conjugate gradient descent**

 * - An ill-conditioned non-quadratic function.

   - |cg_gauss_icond|
 
   - |cg_gauss_icond_conv|

 * - An ill-conditioned very non-quadratic function.

   - |cg_rosen_icond|
 
   - |cg_rosen_icond_conv|

scipy provides :func:`scipy.optimize.minimize` to find the minimum of scalar
functions of one or more variables. The simple conjugate gradient method can
be used by setting the parameter ``method`` to CG ::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> optimize.minimize(f, [2, -1], method="CG")    # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
         fun: 1.6503729082243953e-11
         jac: array([ -6.15347610e-06,   2.53804028e-07])
     message: 'Optimization terminated successfully.'
        nfev: 108
         nit: 13
        njev: 27
      status: 0
     success: True
           x: array([ 0.99999426,  0.99998863])

Gradient methods need the Jacobian (gradient) of the function. They can compute it
numerically, but will perform better if you can pass them the gradient::

    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.fmin_cg(f, [2, 1], fprime=fprime)    # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
         fun: 1.0912121775174348e-11
         jac: array([ -5.25283405e-06,   2.92893689e-07])
     message: 'Optimization terminated successfully.'
        nfev: 27
         nit: 13
        njev: 27
      status: 0
     success: True
           x: array([ 0.99999533,  0.99999081])

Note that the function has only been evaluated 27 times, compared to 108
without the gradient.

Newton and quasi-newton methods
--------------------------------

Newton methods: using the Hessian (2nd differential)
.....................................................

`Newton methods
<http://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>`_ use a
local quadratic approximation to compute the jump direction. For this
purpose, they rely on the 2 first derivative of the function: the
*gradient* and the `Hessian
<http://en.wikipedia.org/wiki/Hessian_matrix>`_.

.. |ncg_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_008.png
   :scale: 90%

.. |ncg_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_108.png
   :scale: 75%

.. |ncg_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_009.png
   :scale: 90%

.. |ncg_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_109.png
   :scale: 75%

.. |ncg_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_010.png
   :scale: 90%

.. |ncg_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_110.png
   :scale: 75%


.. list-table::

 * - **An ill-conditioned quadratic function:**

     Note that, as the quadratic approximation is exact, the Newton
     method is blazing fast

   - |ncg_quad_icond|
 
   - |ncg_quad_icond_conv|

 * - **An ill-conditioned non-quadratic function:**

     Here we are optimizing a Gaussian, which is always below its
     quadratic approximation. As a result, the Newton method overshoots
     and leads to oscillations.

   - |ncg_gauss_icond|
 
   - |ncg_gauss_icond_conv|

 * - **An ill-conditioned very non-quadratic function:**

   - |ncg_rosen_icond|
 
   - |ncg_rosen_icond_conv|

In scipy, you can use the Newton method by setting ``method`` to Newton-CG in
:func:`scipy.optimize.minimize`. Here, CG refers to the fact that an internal
inversion of the Hessian is performed by conjugate gradient ::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.minimize(f, [2,-1], method="Newton-CG", jac=jacobian)    # doctest: +NORMALIZE_WHITESPACE
         fun: 1.5601357400786612e-15
         jac: array([  1.05753092e-07,  -7.48325277e-08])
     message: 'Optimization terminated successfully.'
        nfev: 11
        nhev: 0
         nit: 10
        njev: 52
      status: 0
     success: True
           x: array([ 0.99999995,  0.99999988])

Note that compared to a conjugate gradient (above), Newton's method has
required less function evaluations, but more gradient evaluations, as it
uses it to approximate the Hessian. Let's compute the Hessian and pass it
to the algorithm::

    >>> def hessian(x): # Computed with sympy
    ...     return np.array(((1 - 4*x[1] + 12*x[0]**2, -4*x[0]), (-4*x[0], 2)))
    >>> optimize.minimize(f, [2,-1], method="Newton-CG", jac=fprime, hess=hessian)    # doctest: +NORMALIZE_WHITESPACE
         fun: 1.6277298383706738e-15
         jac: array([  1.11044158e-07,  -7.78093352e-08])
     message: 'Optimization terminated successfully.'
        nfev: 11
        nhev: 10
         nit: 10
        njev: 20
      status: 0
     success: True
           x: array([ 0.99999994,  0.99999988])

.. note:: 
   
    At very high-dimension, the inversion of the Hessian can be costly
    and unstable (large scale > 250).

.. note:: 
   
    Newton optimizers should not to be confused with Newton's root finding
    method, based on the same principles, :func:`scipy.optimize.newton`.

.. _quasi_newton:

Quasi-Newton methods: approximating the Hessian on the fly 
...........................................................

**BFGS**: BFGS (Broyden-Fletcher-Goldfarb-Shanno algorithm) refines at
each step an approximation of the Hessian.

.. |bfgs_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_011.png
   :scale: 90%

.. |bfgs_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_111.png
   :scale: 75%

.. |bfgs_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_012.png
   :scale: 90%

.. |bfgs_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_112.png
   :scale: 75%

.. |bfgs_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_013.png
   :scale: 90%

.. |bfgs_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_113.png
   :scale: 75%


.. list-table::

 * - **An ill-conditioned quadratic function:**

     On a exactly quadratic function, BFGS is not as fast as Newton's
     method, but still very fast.

   - |bfgs_quad_icond|
 
   - |bfgs_quad_icond_conv|

 * - **An ill-conditioned non-quadratic function:**

     Here BFGS does better than Newton, as its empirical estimate of the
     curvature is better than that given by the Hessian.

   - |bfgs_gauss_icond|
 
   - |bfgs_gauss_icond_conv|

 * - **An ill-conditioned very non-quadratic function:**

   - |bfgs_rosen_icond|
 
   - |bfgs_rosen_icond_conv|

::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.minimize(f, [2, -1], method="BFGS", jac=jacobian)    # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
          fun: 2.630637192365927e-16
     hess_inv: array([[ 0.99986623,  2.00004547],
           [ 2.00004547,  4.49857739]])
          jac: array([  6.70894997e-08,  -3.22220060e-08])
      message: 'Optimization terminated successfully.'
         nfev: 10
          nit: 8
         njev: 10
       status: 0
      success: True
            x: array([ 1.        ,  0.99999999])


**L-BFGS:** Limited-memory BFGS Sits between BFGS and conjugate gradient:
in very high dimensions (> 250) the Hessian matrix is too costly to
compute and invert. L-BFGS keeps a low-rank version. In addition, box bounds
are also supported by L-BFGS-B::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> optimize.minimize(f, [2, 2], method="L-BFGS-B", jac=jacobian)    # doctest: +ELLIPSIS
          fun: 1.4417677473011859e-15
     hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
          jac: array([  1.02331202e-07,  -2.59299369e-08])
      message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
         nfev: 17
          nit: 16
       status: 0
      success: True
            x: array([ 1.00000005,  1.00000009]


Gradient-less methods
----------------------

A shooting method: the Powell algorithm
........................................

Almost a gradient approach

.. |powell_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_014.png
   :scale: 90%

.. |powell_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_114.png
   :scale: 75%

.. |powell_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_015.png
   :scale: 90%

.. |powell_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_115.png
   :scale: 75%


.. |powell_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_016.png
   :scale: 90%

.. |powell_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_116.png
   :scale: 75%


.. list-table::

 * - **An ill-conditioned quadratic function:**

     Powell's method isn't too sensitive to local ill-conditionning in
     low dimensions

   - |powell_quad_icond|
 
   - |powell_quad_icond_conv|

 * - **An ill-conditioned very non-quadratic function:**

   - |powell_rosen_icond|
 
   - |powell_rosen_icond_conv|


Simplex method: the Nelder-Mead
................................

The Nelder-Mead algorithms is a generalization of dichotomy approaches to
high-dimensional spaces. The algorithm works by refining a `simplex
<http://en.wikipedia.org/wiki/Simplex>`_, the generalization of intervals
and triangles to high-dimensional spaces, to bracket the minimum. 

**Strong points**: it is robust to noise, as it does not rely on
computing gradients. Thus it can work on functions that are not locally
smooth such as experimental data points, as long as they display a
large-scale bell-shape behavior. However it is slower than gradient-based
methods on smooth, non-noisy functions.

.. |nm_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_017.png
   :scale: 90%

.. |nm_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_117.png
   :scale: 75%


.. |nm_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_018.png
   :scale: 90%

.. |nm_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_118.png
   :scale: 75%


.. list-table::

 * - **An ill-conditioned non-quadratic function:**

   - |nm_gauss_icond|
 
   - |nm_gauss_icond_conv|

 * - **An ill-conditioned very non-quadratic function:**

   - |nm_rosen_icond|
 
   - |nm_rosen_icond_conv|

Using the Nelder-Mead solver in :func:`scipy.optimize.minimize`::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> optimize.minimize(f, [2, -1], method="Nelder-Mead")
     final_simplex: (array([[ 1.00001481,  1.00002828],
           [ 0.9999825 ,  0.9999643 ],
           [ 1.00001465,  1.0000095 ]]), array([  1.11527916e-10,   1.53677305e-10,   4.98835768e-10]))
               fun: 1.1152791599374399e-10
           message: 'Optimization terminated successfully.'
              nfev: 111
               nit: 58
            status: 0
           success: True
                 x: array([ 1.00001481,  1.00002828])


Global optimizers
------------------

If your problem does not admit a unique local minimum (which can be hard
to test unless the function is convex), and you do not have prior
information to initialize the optimization close to the solution, you
may need a global optimizer.

Brute force: a grid search
..........................

:func:`scipy.optimize.brute` evaluates the function on a given grid of
parameters and returns the parameters corresponding to the minimum
value. The parameters are specified with ranges given to
:obj:`numpy.mgrid`. By default, 20 steps are taken in each direction::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> optimize.brute(f, ((-1, 2), (-1, 2)))
    array([ 1.00001462,  1.00001547])


Practical guide to optimization with scipy
===========================================

Choosing a method
------------------

All methods are exposed as the ``method`` argument of
:func:`scipy.optimize.minimize`.

.. image:: auto_examples/images/sphx_glr_plot_compare_optimizers_001.png
   :align: center
   :width: 95%

:Without knowledge of the gradient:

 * In general, prefer **BFGS** or **L-BFGS**, even if you have to approximate
   numerically gradients. These are also the default if you omit the parameter
   ``method`` - depending if the problem has constraints or bounds
 
 * On well-conditioned problems, **Powell**
   and **Nelder-Mead**, both gradient-free methods, work well in
   high dimension, but they collapse for ill-conditioned problems.

:With knowledge of the gradient:

 * **BFGS** or **L-BFGS**.
 
 * Computational overhead of BFGS is larger than that L-BFGS, itself
   larger than that of conjugate gradient. On the other side, BFGS usually
   needs less function evaluations than CG. Thus conjugate gradient method
   is better than BFGS at optimizing computationally cheap functions.
 
:With the Hessian:

 * If you can compute the Hessian, prefer the Newton method
   (**Newton-CG** or **TCG**).

:If you have noisy measurements:

 * Use **Nelder-Mead** or **Powell**.

Making your optimizer faster
-----------------------------

* Choose the right method (see above), do compute analytically the
  gradient and Hessian, if you can.

* Use `preconditionning <http://en.wikipedia.org/wiki/Preconditioner>`_
  when possible.

* Choose your initialization points wisely. For instance, if you are
  running many similar optimizations, warm-restart one with the results of
  another.

* Relax the tolerance if you don't need precision using the parameter ``tol``.

Computing gradients
-------------------

Computing gradients, and even more Hessians, is very tedious but worth
the effort. Symbolic computation with :ref:`Sympy <sympy>` may come in
handy.

.. warning::
   
   A *very* common source of optimization not converging well is human
   error in the computation of the gradient. You can use
   :func:`scipy.optimize.check_grad` to check that your gradient is
   correct. It returns the norm of the different between the gradient
   given, and a gradient computed numerically:

    >>> optimize.check_grad(f, fprime, [2, 2])
    2.384185791015625e-07

   See also :func:`scipy.optimize.approx_fprime` to find your errors.

Synthetic exercices
-------------------

.. |flat_min_0| image:: auto_examples/images/sphx_glr_plot_exercise_flat_minimum_000.png
    :scale: 48%
    :target: auto_examples/plot_exercise_flat_minimum.html

.. |flat_min_1| image:: auto_examples/images/sphx_glr_plot_exercise_flat_minimum_001.png
    :scale: 48%
    :target: auto_examples/plot_exercise_flat_minimum.html

.. image:: auto_examples/images/sphx_glr_plot_exercise_ill_conditioned_001.png
    :scale: 35%
    :target: auto_examples/plot_exercise_ill_conditioned.html
    :align: right

.. topic:: **Exercice: A simple (?) quadratic function**
    :class: green

    Optimize the following function, using K[0] as a starting point::

        np.random.seed(0)
        K = np.random.normal(size=(100, 100))

        def f(x):
            return np.sum((np.dot(K, x - 1))**2) + np.sum(x**2)**2

    Time your approach. Find the fastest approach. Why is BFGS not
    working well?

.. topic:: **Exercice: A locally flat minimum**
    :class: green

    Consider the function `exp(-1/(.1*x**2 + y**2)`. This function admits
    a minimum in (0, 0). Starting from an initialization at (1, 1), try
    to get within 1e-8 of this minimum point.

    .. centered:: |flat_min_0| |flat_min_1|


Special case: non-linear least-squares
========================================

Minimizing the norm of a vector function
-------------------------------------------

Least square problems, minimizing the norm of a vector function, have a
specific structure that can be used in the `Levenberg–Marquardt algorithm
<https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_
implemented in :func:`scipy.optimize.leastsq`.

Lets try to minimize the norm of the following vectorial function::

    >>> def f(x):
    ...     return np.arctan(x) - np.arctan(np.linspace(0, 1, len(x)))

    >>> x0 = np.zeros(10)
    >>> optimize.leastsq(f, x0)
    (array([ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
            0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ]), 2)

This took 67 function evaluations (check it with 'full_output=1'). What
if we compute the norm ourselves and use a good generic optimizer
(BFGS)::

    >>> def g(x):
    ...     return np.sum(f(x)**2)
    >>> optimize.minimize(g, x0, method="BFGS")   #doctest: +ELLIPSIS
      fun: 2.694080799585467e-11
     hess_inv: array([[  1.00000000e+00,   1.51476408e-06,  -9.52938670e-07,
              8.10490662e-07,  -8.70993707e-07,  -2.38913624e-07,
              7.65780553e-08,  -1.22683371e-07,  -2.78964621e-07,
              3.30633449e-07],
           [  1.51476408e-06,   5.22614112e-01,   4.31229250e-03,
             -2.77309751e-03,  -2.20157452e-03,  -2.43622788e-03,
             -5.48382372e-03,  -6.66807125e-04,  -6.81069303e-04,
             -6.35640269e-03],
           [ -9.52938670e-07,   4.31229250e-03,   5.64454479e-01,
              3.44358934e-03,   7.50394924e-03,  -1.94770150e-02,
             -8.05017298e-03,  -7.92117664e-03,  -9.80726517e-03,
              3.50617008e-03],
           [  8.10490662e-07,  -2.77309751e-03,   3.44358934e-03,
              6.24376041e-01,  -5.23908085e-03,   2.09245073e-02,
             -9.17188532e-03,   2.33290841e-03,  -1.94598387e-03,
             -9.59957032e-03],
           [ -8.70993707e-07,  -2.20157452e-03,   7.50394924e-03,
             -5.23908085e-03,   7.06384458e-01,  -2.96921332e-02,
             -7.30974882e-03,  -1.98820791e-02,   4.53601387e-02,
              3.35150772e-03],
           [ -2.38913624e-07,  -2.43622788e-03,  -1.94770150e-02,
              2.09245073e-02,  -2.96921332e-02,   9.82326555e-01,
             -2.74635570e-02,  -7.74369589e-03,   4.05006906e-02,
             -1.09859805e-02],
           [  7.65780553e-08,  -5.48382372e-03,  -8.05017298e-03,
             -9.17188532e-03,  -7.30974882e-03,  -2.74635570e-02,
              9.86851228e-01,   6.99212582e-03,   1.86241008e-02,
              7.19597367e-03],
           [ -1.22683371e-07,  -6.66807125e-04,  -7.92117664e-03,
              2.33290841e-03,  -1.98820791e-02,  -7.74369589e-03,
              6.99212582e-03,   1.05286148e+00,   1.25196160e-01,
             -4.02652002e-02],
           [ -2.78964621e-07,  -6.81069303e-04,  -9.80726517e-03,
             -1.94598387e-03,   4.53601387e-02,   4.05006906e-02,
              1.86241008e-02,   1.25196160e-01,   1.34158407e+00,
             -7.03041201e-02],
           [  3.30633449e-07,  -6.35640269e-03,   3.50617008e-03,
             -9.59957032e-03,   3.35150772e-03,  -1.09859805e-02,
              7.19597367e-03,  -4.02652002e-02,  -7.03041201e-02,
              1.92714740e+00]])
          jac: array([  1.32042630e-10,  -1.57256739e-07,   1.23570681e-06,
            -5.33389992e-07,  -1.53039156e-06,  -3.48637009e-06,
            -4.53222592e-07,  -3.17692260e-06,   4.09300091e-06,
             5.73595621e-07])
      message: 'Optimization terminated successfully.'
         nfev: 144
          nit: 11
         njev: 12
       status: 0
      success: True
            x: array([ -7.38455928e-09,   1.11111023e-01,   2.22222895e-01,
             3.33332997e-01,   4.44443340e-01,   5.55552563e-01,
             6.66666186e-01,   7.77773679e-01,   8.88895440e-01,
             1.00000114e+00])


BFGS needs more function calls, and gives a less precise result.

.. note:: 
   
    `leastsq` is interesting compared to BFGS only if the
    dimensionality of the output vector is large, and larger than the number
    of parameters to optimize.

.. warning::

   If the function is linear, this is a linear-algebra problem, and
   should be solved with :func:`scipy.linalg.lstsq`.

Curve fitting
--------------

.. np.random.seed(0)

.. image:: auto_examples/images/sphx_glr_plot_curve_fit_001.png
    :scale: 48%
    :target: auto_examples/plot_curve_fit.html
    :align: right

.. Comment to make doctest pass
    >>> np.random.seed(0)

Least square problems occur often when fitting a non-linear to data.
While it is possible to construct our optimization problem ourselves,
scipy provides a helper function for this purpose:
:func:`scipy.optimize.curve_fit`::


    >>> def f(t, omega, phi):
    ...     return np.cos(omega * t + phi)
    
    >>> x = np.linspace(0, 3, 50)
    >>> y = f(x, 1.5, 1) + .1*np.random.normal(size=50)

    >>> optimize.curve_fit(f, x, y)
    (array([ 1.51854577,  0.92665541]), array([[ 0.00037994, -0.00056796],
           [-0.00056796,  0.00123978]]))


.. topic:: **Exercise**
   :class: green

   Do the same with omega = 3. What is the difficulty?

Optimization with constraints
==============================

Box bounds
----------

Box bounds correspond to limiting each of the individual parameters of
the optimization. Note that some problems that are not originally written
as box bounds can be rewritten as such via change of variables.

.. image:: auto_examples/images/sphx_glr_plot_constraints_002.png
    :target: auto_examples/plot_constraints.html
    :align: right
    :scale: 75%

* :func:`scipy.optimize.fminbound` for 1D-optimization
* :func:`scipy.optimize.fmin_l_bfgs_b` a 
  :ref:`quasi-Newton <quasi_newton>` method with bound constraints::

    >>> def f(x):
    ...    return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)
    >>> optimize.fmin_l_bfgs_b(f, np.array([0, 0]), approx_grad=1, bounds=((-1.5, 1.5), (-1.5, 1.5)))   # doctest: +ELLIPSIS
    (array([ 1.5,  1.5]), 1.5811388300841898, {...})


General constraints
--------------------

Equality and inequality constraints specified as functions: `f(x) = 0`
and `g(x)< 0`.

* :func:`scipy.optimize.fmin_slsqp` Sequential least square programming:
  equality and inequality constraints:

  .. image:: auto_examples/images/sphx_glr_plot_non_bounds_constraints_001.png
    :target: auto_examples/plot_non_bounds_constraints.html
    :align: right
    :scale: 75%

  ::

    >>> def f(x):
    ...     return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)

    >>> def constraint(x):
    ...     return np.atleast_1d(1.5 - np.sum(np.abs(x)))

    >>> optimize.fmin_slsqp(f, np.array([0, 0]), ieqcons=[constraint, ]) #doctest: +ELLIPSIS
    Optimization terminated successfully.    (Exit mode 0)
                Current function value: 2.4748737350...
                Iterations: 5
                Function evaluations: 20
                Gradient evaluations: 5
    array([ 1.25004696,  0.24995304])



* :func:`scipy.optimize.fmin_cobyla` Constraints optimization by linear 
  approximation: inequality constraints only::

    >>> optimize.fmin_cobyla(f, np.array([0, 0]), cons=constraint)
    array([ 1.25009622,  0.24990378])

.. warning:: 
   
   The above problem is known as the `Lasso
   <http://en.wikipedia.org/wiki/Lasso_(statistics)#LASSO_method>`_
   problem in statistics, and there exists very efficient solvers for it
   (for instance in `scikit-learn <http://scikit-learn.org>`_). In
   general do not use generic solvers when specific ones exist.

.. topic:: **Lagrange multipliers**

   If you are ready to do a bit of math, many constrained optimization
   problems can be converted to non-constrained optimization problems
   using a mathematical trick known as `Lagrange multipliers
   <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_.
   
