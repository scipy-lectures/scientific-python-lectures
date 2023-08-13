..
  For doctesting
  >>> import numpy as np

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

   .. rst-class:: horizontal

    * :ref:`NumPy <numpy>`
    * :ref:`SciPy <scipy>`
    * :ref:`Matplotlib <matplotlib>`

.. seealso::  **References**

   Mathematical optimization is very ... mathematical. If you want
   performance, it really pays to read the books:

   * `Convex Optimization <https://web.stanford.edu/~boyd/cvxbook/>`_
     by Boyd and Vandenberghe (pdf available free online).

   * `Numerical Optimization
     <https://users.eecs.northwestern.edu/~nocedal/book/num-opt.html>`_,
     by Nocedal and Wright. Detailed reference on gradient descent methods.

   * `Practical Methods of Optimization
     <https://www.amazon.com/gp/product/0471494631/ref=ox_sc_act_title_1?ie=UTF8&smid=ATVPDKIKX0DER>`_ by Fletcher: good at hand-waving explanations.

.. include:: ../../includes/big_toc_css.rst
   :start-line: 1


.. contents:: Chapters contents
   :local:
   :depth: 2

.. XXX: should I discuss root finding?


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
`Linear Programming <https://en.wikipedia.org/wiki/Linear_programming>`_
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

    >>> import numpy as np
    >>> import scipy as sp
    >>> def f(x):
    ...     return -np.exp(-(x - 0.5)**2)
    >>> result = sp.optimize.minimize_scalar(f)
    >>> result.success # check if solver was successful
    True
    >>> x_min = result.x
    >>> x_min
    0.50...
    >>> x_min - 0.5
    5.8...e-09


.. |1d_optim_1| image:: auto_examples/images/sphx_glr_plot_1d_optim_001.png
   :scale: 90%

.. |1d_optim_2| image:: auto_examples/images/sphx_glr_plot_1d_optim_002.png
   :scale: 75%

.. |1d_optim_3| image:: auto_examples/images/sphx_glr_plot_1d_optim_003.png
   :scale: 90%

.. |1d_optim_4| image:: auto_examples/images/sphx_glr_plot_1d_optim_004.png
   :scale: 75%

.. list-table:: **Brent's method on a quadratic function**: it
                converges in 3 iterations, as the quadratic
                approximation is then exact.

   * - |1d_optim_1|

     - |1d_optim_2|

.. list-table:: **Brent's method on a non-convex function**: note that
                the fact that the optimizer avoided the local minimum
                is a matter of luck.

   * - |1d_optim_3|

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

`Gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_
basically consists in taking small steps in the direction of the
gradient, that is the direction of the *steepest descent*.

.. |gradient_quad_cond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_001.png
   :scale: 90%

.. |gradient_quad_cond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_020.png
   :scale: 75%

.. |gradient_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_003.png
   :scale: 90%

.. |gradient_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_022.png
   :scale: 75%

.. list-table:: **Fixed step gradient descent**
   :widths: 1 1 1

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
<https://en.wikipedia.org/wiki/Condition_number>`_) functions are harder
to optimize.

.. topic:: **Take home message: conditioning number and preconditioning**

   If you know natural scaling for your variables, prescale them so that
   they behave similarly. This is related to `preconditioning
   <https://en.wikipedia.org/wiki/Preconditioner>`_.

Also, it clearly can be advantageous to take bigger steps. This
is done in gradient descent code using a
`line search <https://en.wikipedia.org/wiki/Line_search>`_.

.. |agradient_quad_cond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_002.png
   :scale: 90%

.. |agradient_quad_cond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_021.png
   :scale: 75%

.. |agradient_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_004.png
   :scale: 90%

.. |agradient_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_023.png
   :scale: 75%

.. |agradient_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_005.png
   :scale: 90%

.. |agradient_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_024.png
   :scale: 75%

.. |agradient_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_006.png
   :scale: 90%

.. |agradient_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_025.png
   :scale: 75%


.. list-table:: **Adaptive step gradient descent**
   :widths: 1 1 1

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

.. |cg_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_007.png
   :scale: 90%

.. |cg_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_026.png
   :scale: 75%

.. |cg_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_008.png
   :scale: 90%

.. |cg_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_027.png
   :scale: 75%


.. list-table:: **Conjugate gradient descent**
   :widths: 1 1 1

   * - An ill-conditioned non-quadratic function.

     - |cg_gauss_icond|

     - |cg_gauss_icond_conv|

   * - An ill-conditioned very non-quadratic function.

     - |cg_rosen_icond|

     - |cg_rosen_icond_conv|

SciPy provides :func:`scipy.optimize.minimize` to find the minimum of scalar
functions of one or more variables. The simple conjugate gradient method can
be used by setting the parameter ``method`` to CG ::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> sp.optimize.minimize(f, [2, -1], method="CG")
     message: Optimization terminated successfully.
     success: True
      status: 0
         fun: 1.650...e-11
           x: [ 1.000e+00  1.000e+00]
         nit: 13
         jac: [-6.15...e-06  2.53...e-07]
        nfev: 81
        njev: 27

Gradient methods need the Jacobian (gradient) of the function. They can compute it
numerically, but will perform better if you can pass them the gradient::

    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> sp.optimize.minimize(f, [2, 1], method="CG", jac=jacobian)
     message: Optimization terminated successfully.
     success: True
      status: 0
         fun: 2.95786...e-14
           x: [ 1.000e+00  1.000e+00]
         nit: 8
         jac: [ 7.183e-07 -2.990e-07]
        nfev: 16
        njev: 16

Note that the function has only been evaluated 27 times, compared to 108
without the gradient.

Newton and quasi-newton methods
--------------------------------

Newton methods: using the Hessian (2nd differential)
.....................................................

`Newton methods
<https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>`_ use a
local quadratic approximation to compute the jump direction. For this
purpose, they rely on the 2 first derivative of the function: the
*gradient* and the `Hessian
<https://en.wikipedia.org/wiki/Hessian_matrix>`_.

.. |ncg_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_009.png
   :scale: 90%

.. |ncg_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_028.png
   :scale: 75%

.. |ncg_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_010.png
   :scale: 90%

.. |ncg_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_029.png
   :scale: 75%

.. |ncg_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_011.png
   :scale: 90%

.. |ncg_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_030.png
   :scale: 75%


.. list-table::
   :widths: 1 1 1

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

In SciPy, you can use the Newton method by setting ``method`` to Newton-CG in
:func:`scipy.optimize.minimize`. Here, CG refers to the fact that an internal
inversion of the Hessian is performed by conjugate gradient ::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> sp.optimize.minimize(f, [2,-1], method="Newton-CG", jac=jacobian)
     message: Optimization terminated successfully.
     success: True
      status: 0
         fun: 1.5601357400786612e-15
           x: [ 1.000e+00  1.000e+00]
         nit: 10
         jac: [ 1.058e-07 -7.483e-08]
        nfev: 11
        njev: 33
        nhev: 0

Note that compared to a conjugate gradient (above), Newton's method has
required less function evaluations, but more gradient evaluations, as it
uses it to approximate the Hessian. Let's compute the Hessian and pass it
to the algorithm::

    >>> def hessian(x): # Computed with sympy
    ...     return np.array(((1 - 4*x[1] + 12*x[0]**2, -4*x[0]), (-4*x[0], 2)))
    >>> sp.optimize.minimize(f, [2,-1], method="Newton-CG", jac=jacobian, hess=hessian)
     message: Optimization terminated successfully.
     success: True
      status: 0
         fun: 1.6277298383706738e-15
           x: [ 1.000e+00  1.000e+00]
         nit: 10
         jac: [ 1.110e-07 -7.781e-08]
        nfev: 11
        njev: 11
        nhev: 10

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

.. |bfgs_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_012.png
   :scale: 90%

.. |bfgs_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_031.png
   :scale: 75%

.. |bfgs_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_013.png
   :scale: 90%

.. |bfgs_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_032.png
   :scale: 75%

Full code examples
==================

.. include the gallery. Skip the first line to avoid the "orphan"
   declaration

.. include:: auto_examples/index.rst
   :start-line: 1


.. |bfgs_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_014.png
   :scale: 90%

.. |bfgs_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_033.png
   :scale: 75%


.. list-table::
   :widths: 1 1 1

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
    >>> sp.optimize.minimize(f, [2, -1], method="BFGS", jac=jacobian)
     message: Optimization terminated successfully.
     success: True
      status: 0
         fun: 2.630637192365927e-16
           x: [ 1.000e+00  1.000e+00]
         nit: 8
         jac: [ 6.709e-08 -3.222e-08]
    hess_inv: [[ 9.999e-01  2.000e+00]
               [ 2.000e+00  4.499e+00]]
        nfev: 10
        njev: 10

**L-BFGS:** Limited-memory BFGS Sits between BFGS and conjugate gradient:
in very high dimensions (> 250) the Hessian matrix is too costly to
compute and invert. L-BFGS keeps a low-rank version. In addition, box bounds
are also supported by L-BFGS-B::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> def jacobian(x):
    ...     return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    >>> sp.optimize.minimize(f, [2, 2], method="L-BFGS-B", jac=jacobian)
     message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
     success: True
      status: 0
         fun: 1.4417677473...e-15
           x: [ 1.000e+00  1.000e+00]
         nit: 16
         jac: [ 1.023e-07 -2.593e-08]
        nfev: 17
        njev: 17
    hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

Gradient-less methods
----------------------

A shooting method: the Powell algorithm
........................................

Almost a gradient approach

.. |powell_quad_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_015.png
   :scale: 90%

.. |powell_quad_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_034.png
   :scale: 75%

.. |powell_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_016.png
   :scale: 90%

.. |powell_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_035.png
   :scale: 75%


.. |powell_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_017.png
   :scale: 90%

.. |powell_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_036.png
   :scale: 75%


.. list-table::
   :widths: 1 1 1

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
<https://en.wikipedia.org/wiki/Simplex>`_, the generalization of intervals
and triangles to high-dimensional spaces, to bracket the minimum.

**Strong points**: it is robust to noise, as it does not rely on
computing gradients. Thus it can work on functions that are not locally
smooth such as experimental data points, as long as they display a
large-scale bell-shape behavior. However it is slower than gradient-based
methods on smooth, non-noisy functions.

.. |nm_gauss_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_018.png
   :scale: 90%

.. |nm_gauss_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_037.png
   :scale: 75%


.. |nm_rosen_icond| image:: auto_examples/images/sphx_glr_plot_gradient_descent_019.png
   :scale: 90%

.. |nm_rosen_icond_conv| image:: auto_examples/images/sphx_glr_plot_gradient_descent_038.png
   :scale: 75%


.. list-table::
   :widths: 1 1 1

   * - **An ill-conditioned non-quadratic function:**

     - |nm_gauss_icond|

     - |nm_gauss_icond_conv|

   * - **An ill-conditioned very non-quadratic function:**

     - |nm_rosen_icond|

     - |nm_rosen_icond_conv|

Using the Nelder-Mead solver in :func:`scipy.optimize.minimize`::

    >>> def f(x):   # The rosenbrock function
    ...     return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
    >>> sp.optimize.minimize(f, [2, -1], method="Nelder-Mead")
           message: Optimization terminated successfully.
           success: True
            status: 0
               fun: 1.11527915993744e-10
                 x: [ 1.000e+00  1.000e+00]
               nit: 58
              nfev: 111
     final_simplex: (array([[ 1.000e+00,  1.000e+00],
                           [ 1.000e+00,  1.000e+00],
                           [ 1.000e+00,  1.000e+00]]), array([ 1.115e-10,  1.537e-10,  4.988e-10]))

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
    >>> sp.optimize.brute(f, ((-1, 2), (-1, 2))) # doctest:   +ELLIPSIS
    array([1.0000...,  1.0000...])


Practical guide to optimization with SciPy
==========================================

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

* Use `preconditionning <https://en.wikipedia.org/wiki/Preconditioner>`_
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

    >>> sp.optimize.check_grad(f, jacobian, [2, -1])
    2.384185791015625e-07

   See also :func:`scipy.optimize.approx_fprime` to find your errors.

Synthetic exercices
-------------------

.. |flat_min_0| image:: auto_examples/images/sphx_glr_plot_exercise_flat_minimum_001.png
    :scale: 48%
    :target: auto_examples/plot_exercise_flat_minimum.html

.. |flat_min_1| image:: auto_examples/images/sphx_glr_plot_exercise_flat_minimum_002.png
    :scale: 48%
    :target: auto_examples/plot_exercise_flat_minimum.html

.. image:: auto_examples/images/sphx_glr_plot_exercise_ill_conditioned_001.png
    :scale: 35%
    :target: auto_examples/plot_exercise_ill_conditioned.html
    :align: right

.. topic:: **Exercice: A simple (?) quadratic function**
    :class: green

    Optimize the following function, using K[0] as a starting point::

        rng = np.random.default_rng(27446968)
        K = rng.normal(size=(100, 100))

        def f(x):
            return np.sum((K @ (x - 1))**2) + np.sum(x**2)**2

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
    >>> sp.optimize.leastsq(f, x0)
    (array([0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
            0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ]), 2)

This took 67 function evaluations (check it with 'full_output=1'). What
if we compute the norm ourselves and use a good generic optimizer
(BFGS)::

    >>> def g(x):
    ...     return np.sum(f(x)**2)
    >>> result = sp.optimize.minimize(g, x0, method="BFGS")
    >>> result.fun
    2.6940...e-11

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

.. image:: auto_examples/images/sphx_glr_plot_curve_fitting_001.png
    :scale: 48%
    :target: auto_examples/plot_curve_fitting.html
    :align: right

Least square problems occur often when fitting a non-linear to data.
While it is possible to construct our optimization problem ourselves,
SciPy provides a helper function for this purpose:
:func:`scipy.optimize.curve_fit`::


    >>> def f(t, omega, phi):
    ...     return np.cos(omega * t + phi)

    >>> x = np.linspace(0, 3, 50)
    >>> rng = np.random.default_rng(27446968)
    >>> y = f(x, 1.5, 1) + .1*rng.normal(size=50)

    >>> sp.optimize.curve_fit(f, x, y)
    (array([1.4812..., 0.9999...]), array([[ 0.0003..., -0.0004...],
           [-0.0004...,  0.0010...]]))


.. topic:: **Exercise**
   :class: green

   Do the same with omega = 3. What is the difficulty?

Optimization with constraints
==============================

Box bounds
----------

Box bounds correspond to limiting each of the individual parameters of
the optimization. Note that some problems that are not originally written
as box bounds can be rewritten as such via change of variables. Both
:func:`scipy.optimize.minimize_scalar` and :func:`scipy.optimize.minimize`
support bound constraints with the parameter ``bounds``::

    >>> def f(x):
    ...    return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)
    >>> sp.optimize.minimize(f, np.array([0, 0]), bounds=((-1.5, 1.5), (-1.5, 1.5)))
      message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
      success: True
       status: 0
          fun: 1.5811388300841898
            x: [ 1.500e+00  1.500e+00]
          nit: 2
          jac: [-9.487e-01 -3.162e-01]
         nfev: 9
         njev: 3
         hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>

.. image:: auto_examples/images/sphx_glr_plot_constraints_002.png
    :target: auto_examples/plot_constraints.html
    :align: right
    :scale: 75%


General constraints
--------------------

Equality and inequality constraints specified as functions: :math:`f(x) = 0`
and :math:`g(x) < 0`.

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

    >>> x0 = np.array([0, 0])
    >>> sp.optimize.minimize(f, x0, constraints={"fun": constraint, "type": "ineq"})
     message: Optimization terminated successfully
     success: True
      status: 0
         fun: 2.4748737350439685
           x: [ 1.250e+00  2.500e-01]
         nit: 5
         jac: [-7.071e-01 -7.071e-01]
        nfev: 15
        njev: 5

.. warning::

   The above problem is known as the `Lasso
   <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
   problem in statistics, and there exist very efficient solvers for it
   (for instance in `scikit-learn <https://scikit-learn.org>`_). In
   general do not use generic solvers when specific ones exist.

.. topic:: **Lagrange multipliers**

   If you are ready to do a bit of math, many constrained optimization
   problems can be converted to non-constrained optimization problems
   using a mathematical trick known as `Lagrange multipliers
   <https://en.wikipedia.org/wiki/Lagrange_multiplier>`_.

Full code examples
==================

.. include the gallery. Skip the first line to avoid the "orphan"
   declaration

.. include:: auto_examples/index.rst
    :start-line: 1

.. seealso::  **Other Software**

    SciPy tries to include the best well-established, general-use,
    and permissively-licensed optimization algorithms available. However,
    even better options for a given task may be available in other libraries;
    please also see IPOPT_ and PyGMO_.

.. _IPOPT: https://github.com/xuy/pyipopt
.. _PyGMO: https://esa.github.io/pygmo2/
