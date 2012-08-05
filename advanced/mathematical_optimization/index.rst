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

Smooth and non-smooth problems
-------------------------------

Noisy versus exact cost functions
----------------------------------

Constraints
------------

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

