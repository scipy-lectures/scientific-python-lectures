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

.. currentmodule:: scipy.optimize

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

The :func:`brent` can be used for efficient minimization of 1D functions.
It combines a bracketing strategy with a parabolic approximation.

.. |1d_optim_1| image:: auto_examples/images/plot_1d_optim_1.png

.. |1d_optim_2| image:: auto_examples/images/plot_1d_optim_2.png
   :scale: 83%

.. |1d_optim_3| image:: auto_examples/images/plot_1d_optim_3.png

.. |1d_optim_4| image:: auto_examples/images/plot_1d_optim_4.png
   :scale: 83%

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
    0.6999999997839...
    >>> x_min - .7 #doctest: +ELLIPSIS
    -2.1605...e-10

.. note:: 
   
   Brent's method can be used for optimization constraint to an
   intervale using :func:`fminbound`

.. note::
   
   In scipy 0.11, :func:`minimize_scalar` gives a generic
   interface to 1D scalar minimization


Gradient and conjugate gradient methods
========================================

.. |gradient_quad_cond| image:: auto_examples/images/plot_gradient_descent_0.png

.. |gradient_quad_cond_conv| image:: auto_examples/images/plot_gradient_descent_10.png
   :scale: 83%

.. |gradient_quad_icond| image:: auto_examples/images/plot_gradient_descent_1.png

.. |gradient_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_11.png
   :scale: 83%


.. list-table:: **Fixed step gradient descent**

 * - A well-conditionned quadratic function.

   - |gradient_quad_cond|
 
   - |gradient_quad_cond_conv|

 * - An ill-conditionned quadratic function.

   - |gradient_quad_icond|
 
   - |gradient_quad_icond_conv|


.. |agradient_quad_cond| image:: auto_examples/images/plot_gradient_descent_100.png

.. |agradient_quad_cond_conv| image:: auto_examples/images/plot_gradient_descent_110.png
   :scale: 83%

.. |agradient_quad_icond| image:: auto_examples/images/plot_gradient_descent_101.png

.. |agradient_quad_icond_conv| image:: auto_examples/images/plot_gradient_descent_111.png
   :scale: 83%

.. |agradient_gauss_icond| image:: auto_examples/images/plot_gradient_descent_103.png

.. |agradient_gauss_icond_conv| image:: auto_examples/images/plot_gradient_descent_113.png
   :scale: 83%

.. |agradient_rosen_icond| image:: auto_examples/images/plot_gradient_descent_104.png

.. |agradient_rosen_icond_conv| image:: auto_examples/images/plot_gradient_descent_114.png
   :scale: 83%


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


Newton and quasy-newton methods
================================

Special case: least-squares
============================

linalg.leastsq

optimize.curve_fit

Simplex methods
================

Alternate optimization: block coordinate methods
=================================================

Global optimizers
==================

Optimization with constraints
==============================

