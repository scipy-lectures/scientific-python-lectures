==========================
Mathematical optimization
==========================

:authors: Gaël Varoquaux

`Mathematical optimization
<http://en.wikipedia.org/wiki/Mathematical_optimization>`_ deals with the
problem of finding numerically minimums (or maximums or zeros) of
a function. In this context, the function is called *cost function*, or
*objective function*, or *energy*.

.. topic:: Prerequisites

    * Numpy, Scipy
    * IPython
    * matplotlib

.. contents:: Chapters contents
   :local:
   :depth: 4


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


Special case: least-squares
============================

Gradient and conjugate gradient methods
========================================

Newton and quasy-newton methods
================================

Simplex methods
================

Alternate optimization: block coordinate methods
=================================================

Global optimizers
==================

Optimization with constraints
==============================

