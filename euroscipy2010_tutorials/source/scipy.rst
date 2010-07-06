Scipy : high-level scientific computing
=========================================

.. topic:: Scipy

    The ``scipy`` package contains various toolboxes dedicated to common
    issues in scientific computing. Its different submodules correspond
    to different applications, such as interpolation, integration,
    optimization, image processing, statistics, special functions, etc.

    ``scipy`` can be compared to other standard scientific-computing
    libraries, such as the GSL (GNU Scientific  Library for C and C++),
    or Matlab's toolboxes. ``scipy`` is the core package for scientific
    routines in Python; it is meant to operate efficiently on ``numpy``
    arrays, so that numpy and scipy work hand in hand.

    Before implementing a routine, if is worth checking if the desired
    data processing is not already implemented in Scipy. As
    non-professional programmers, scientists often tend to **re-invent the
    wheel**, which leads to buggy, non-optimal, difficult-to-share and
    unmaintainable code. By contrast, ``Scipy``'s routines are optimized
    and tested, and should therefore be used when possible.


.. warning::

    This tutorial is far from an introduction to numerical computing.
    As enumerating the different submodules and functions in scipy would
    be very boring, we concentrate instead on a few examples to give a
    general idea of how to use ``scipy`` for scientific computing.

To begin with ::

    >>> import numpy as np
    >>> import scipy

Scipy builds upon Numpy
-------------------------

Numpy is required for running Scipy but also for using it. The most
important type introduced to Python is the N dimensional array,
and it can be seen that Scipy uses the same::

    >>> scipy.ndarray is np.ndarray
    True

Moreover most of the Scipy usual functions are provided by Numpy::

    >>> scipy.cos is np.cos
    True

If you would like to know the objects used from Numpy, have a look at
the  ``scipy.__file__[:-1]`` file. On version '0.6.0', the whole Numpy
namespace is imported by the line ``from numpy import *``.

Signal processing
==================

* Detrend:

  .. plot:: demo_detrend.py
    :include-source:
    :hide-links:


Linear algebra operations with ``scipy.linalg``
-----------------------------------------------
First, the linalg module provides standard linear algebra operations.
The ``det`` function computes the determinant of a square matrix::

    >>> from scipy import linalg
    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> linalg.det(arr)
    -2.0
    >>> arr = np.array([[3, 2],
    ...                 [6, 4]])
    >>> linalg.det(arr)
    0.0
    >>> linalg.det(np.ones((3, 4)))
    Traceback (most recent call last):
    ...
    ValueError: expected square matrix

The ``inv`` function computes the inverse of a square matrix::

    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> iarr = linalg.inv(arr)
    >>> iarr
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.allclose(np.dot(arr, iarr), np.eye(2))
    True

Note that in case you use the matrix type, the inverse is computed when
requesting the ``I`` attribute::

    >>> ma = np.matrix(arr, copy=False)
    >>> np.allclose(ma.I, iarr)
    True

Finally computing the inverse of a singular matrix (its determinant is zero)
will raise ``LinAlgError``::

    >>> arr = np.array([[3, 2],
    ...                 [6, 4]])
    >>> linalg.inv(arr)
    Traceback (most recent call last):
    ...
    LinAlgError: singular matrix

More advanced operations are available like singular-value decomposition
(SVD)::

    >>> arr = np.arange(12).reshape((3, 4)) + 1
    >>> uarr, spec, vharr = linalg.svd(arr)

The resulting array spectrum is::

    >>> spec
    array([  2.54368356e+01,   1.72261225e+00,   5.14037515e-16])

For the recomposition, an alias for manipulating matrix will first
be defined::

    >>> asmat = np.asmatrix

then the steps are::

    >>> sarr = np.zeros((3, 4))
    >>> sarr.put((0, 5, 10), spec)
    >>> svd_mat = asmat(uarr) * asmat(sarr) * asmat(vharr)
    >>> np.allclose(svd_mat, arr)
    True

SVD is commonly used in statistics or signal processing.  Many other
standard decompositions (QR, LU, Cholesky, Schur), as well as solvers
for linear systems, are available in ``scipy.linalg``.

Numerical integration with ``scipy.integrate``
----------------------------------------------
The most generic integration routine is ``scipy.integrate.quad``::

    >>> from scipy.integrate import quad
    >>> res, err = quad(np.sin, 0, np.pi/2)
    >>> np.allclose(res, 1)
    True
    >>> np.allclose(err, 1 - res)
    True

Others integration schemes are available with ``fixed_quad``,
``quadrature``, ``romberg``.

``scipy.integrate`` also features routines for Ordinary differential
equations (ODE) integration. In particular, ``scipy.integrate.odeint``
is a general-purpose integrator using LSODA (Livermore solver for
ordinary differential equations with automatic method switching
for stiff and nonstiff problems), see the `ODEPACK Fortran library`_
for more details.

.. _`ODEPACK Fortran library` : http://people.sc.fsu.edu/~jburkardt/f77_src/odepack/odepack.html

``odeint`` solves first-order ODE systems of the form::

``dy/dt = rhs(y1, y2, .., t0,...)``

As an introduction, let us solve the ODE ``dy/dt = -2y`` between ``t =
0..4``, with the  initial condition ``y(t=0) = 1``. First the function
computing the derivative of the position needs to be defined::

    >>> def calc_derivative(ypos, time, counter_arr):
    ...     counter_arr += 1
    ...     return -2*ypos
    ...

An extra argument ``counter_arr`` has been added to illustrate that the
function may be called several times for a single time step, until solver
convergence. The counter array is defined as::

    >>> counter = np.zeros((1,), np.uint16)

The trajectory will now be computed::

    >>> from scipy.integrate import odeint
    >>> time_vec = np.linspace(0, 4, 40)
    >>> yvec, info = odeint(calc_derivative, 1, time_vec,
    ...                     args=(counter,), full_output=True)

Thus the derivative function has been called more than 40 times::

    >>> counter
    array([129], dtype=uint16)

and the cumulative iterations number for the 10 first convergences
can be obtained by::

    >>> info['nfe'][:10]
    array([31, 35, 43, 49, 53, 57, 59, 63, 65, 69], dtype=int32)

The solver requires more iterations at start. The final trajectory is
seen on the Matplotlib figure computed with odeint-introduction.py_.

.. image:: odeint-introduction.png
   :align: center

.. _odeint-introduction.py : data/odeint-introduction.py

Another example with ``odeint`` will be a damped spring-mass oscillator
(2nd order oscillator). The position of a mass attached to a spring obeys
the 2nd order ODE ``y'' + 2 eps wo  y' + wo^2 y = 0`` with ``wo^2 = k/m``
being ``k`` the spring constant, ``m`` the mass and ``eps=c/(2 m wo)``
with ``c`` the damping coefficient.
For a computing example, the parameters will be::

    >>> mass = 0.5 # kg
    >>> kspring = 4 # N/m
    >>> cviscous = 0.4 # N s/m

so the system will be underdamped because::

    >>> eps = cviscous / (2 * mass * np.sqrt(kspring/mass))
    >>> eps < 1
    True

For the ``odeint`` solver the 2nd order equation needs to be transformed in a
system of two first-order equations for the vector ``Y=(y, y')``.  It will
be convenient to define ``nu = 2 eps wo = c / m`` and ``om = wo^2 = k/m``::

    >>> nu_coef = cviscous/mass
    >>> om_coef = kspring/mass

Thus the function will calculate the velocity and acceleration by::

    >>> def calc_deri(yvec, time, nuc, omc):
    ...     return (yvec[1], -nuc * yvec[1] - omc * yvec[0])
    ...
    >>> time_vec = np.linspace(0, 10, 100)
    >>> yarr = odeint(calc_deri, (1, 0), time_vec, args=(nu_coef, om_coef))

The final position and velocity are shown on a Matplotlib figure
built with the odeint-damped-spring-mass.py_ script.

.. image:: odeint-damped-spring-mass.png
   :align: center

.. _odeint-damped-spring-mass.py: data/odeint-damped-spring-mass.py

There is no Partial Differential Equations (PDE) solver
in scipy. Some PDE packages are written in Python, such
as fipy_ or SfePy_.

.. _fipy: http://www.ctcms.nist.gov/fipy/
.. _SfePy: http://code.google.com/p/sfepy/

Fast Fourier transforms with ``scipy.fftpack``
----------------------------------------------
The ``fftpack`` module allows to compute fast Fourier transforms.
As an illustration, an input signal may look like::

    >>> time_step = 0.1
    >>> period = 5.
    >>> time_vec = np.arange(0, 20, time_step)
    >>> sig = np.sin(2 * np.pi / period * time_vec) + \
    ...       np.cos(10 * np.pi * time_vec)

However the observer does not know the signal frequency, only
the sampling time step of the signal ``sig``. But the signal
is supposed to come from a real function so the Fourier transform
will be symmetric.
The ``fftfreq`` function will generate the sampling frequencies and
``fft`` will compute the fast fourier transform::

    >>> from scipy import fftpack
    >>> sample_freq = fftpack.fftfreq(sig.size, d=time_step)
    >>> sig_fft = fftpack.fft(sig)

Nevertheless only the positive part will be used for finding the frequency
because the resulting power is symmetric::

    >>> pidxs = np.where(sample_freq > 0)
    >>> freqs = sample_freq[pidxs]
    >>> power = np.abs(sig_fft)[pidxs]

.. image:: fftpack-frequency.png
   :align: center

Thus the signal frequency can be found by::

    >>> freq = freqs[power.argmax()]
    >>> np.allclose(freq, 1./period)
    True

Now only the main signal component will be extracted from the
Fourier transform::

    >>> sig_fft[np.abs(sample_freq) > freq] = 0

The resulting signal can be computed by the ``ifft`` function::

    >>> main_sig = fftpack.ifft(sig_fft)

The result is shown on the Matplotlib figure generated by the
fftpack-illustration.py_ script.

.. image:: fftpack-signals.png
   :align: center

.. _fftpack-illustration.py: data/fftpack-illustration.py


Interpolation: ``scipy.interpolate``
------------------------------------
The ``scipy.interpolate`` is useful for fitting a function from experimental
data and thus evaluating points where no measure exists. The module is based
on the `FITPACK Fortran subroutines`_ from the netlib_ project.

.. _`FITPACK Fortran subroutines` : http://www.netlib.org/dierckx/index.html
.. _netlib : http://www.netlib.org

By imagining experimental data close to a sinus function::

    >>> measured_time = np.linspace(0, 1, 10)
    >>> noise = (np.random.random(10)*2 - 1) * 1e-1
    >>> measures = np.sin(2 * np.pi * measured_time) + noise

The ``interp1d`` class can built a linear interpolation function::

    >>> from scipy.interpolate import interp1d
    >>> linear_interp = interp1d(measured_time, measures)

Then the ``linear_interp`` instance needs to be evaluated on time of
interest::

    >>> computed_time = np.linspace(0, 1, 50)
    >>> linear_results = linear_interp(computed_time)

A cubic interpolation can also be selected by providing the ``kind`` optional
keyword argument::

    >>> cubic_interp = interp1d(measured_time, measures, kind='cubic')
    >>> cubic_results = cubic_interp(computed_time)

The results are now gathered on a Matplotlib figure generated by
the script scipy-interpolation.py_.

.. image:: interpolation.png
   :align: center

.. _scipy-interpolation.py : data/scipy-interpolation.py

``scipy.interpolate.interp2d`` is similar to ``interp1d``, but for 2-D
arrays. Note that for the ``interp`` family, the computed time must stay
within the measured time range. See the summary exercice  on `Maximum
wind speed prediction at the Sprogø station`_ for a more advance spline
interpolation example.


Optimization and fit: ``scipy.optimize``
----------------------------------------

The ``scipy.optimize`` module provides useful algorithms for function
minimization (scalar or multi-dimensional), curve fitting and root finding.


**Example: Minimizing a scalar function using different algorithms**

Let's define the following function: ::

    >>> def f(x): return x**2 + 10*np.sin(x)

and plot it:

.. doctest::

    >>> x = np.arange(-10,10,0.1)
    >>> plt.plot(x, f(x)) # doctest:+SKIP
    >>> plt.show() # doctest:+SKIP

.. image:: minima-function.png
   :align: center

This function has a global minimum around -1.3 and a local minimum around 3.8.


To find the global minimum, the simplest algorithm is the brute force algorithm,
in which the function is evaluated on each point of a given grid: ::

    >>> from scipy import optimize
    >>> grid = (-10, 10, 0.1)
    >>> optimize.brute(f, (grid,))
    array([-1.30641113])

This simple alorithm becomes very slow as the size of the grid grows, so you
should use ``optimize.brent`` instead for scalar functions: ::

    >>> optimize.brent(f)
    -1.3064400120612139

To find the local minimum, let's add some constraints on the variable using
``optimize.fminbound``: ::

    >>> # search the minimum only between 0 and 10
    >>> optimize.fminbound(f, 0, 10)
    array([ 3.83746712])

You can find algorithms with the same functionalities for multi-dimensional
problems in ``scipy.optimize``.


See the summary exercise on :ref:`summary_exercise_optimize` for a
more advanced example.





Image processing: ``scipy.ndimage``
-----------------------------------

.. include:: image_processing/image_processing.rst





Summary exercices on scientific computing
-----------------------------------------

The summary exercices use mainly Numpy, Scipy and Matplotlib. They first aim at
providing real life examples on scientific computing with Python. Once the
groundwork is introduced, the interested user is invited to try some exercices.

.. only:: latex

    .. toctree::
       :maxdepth: 1

       summary-exercices/stats-interpolate.rst
       summary-exercices/optimize-fit.rst
       summary-exercices/image-processing.rst
       summary-exercices/answers_image_processing.rst

.. only:: html

   Exercises:

   .. toctree::
       :maxdepth: 1

       summary-exercices/stats-interpolate.rst
       summary-exercices/optimize-fit.rst
       summary-exercices/image-processing.rst

   Proposed solutions:

   .. toctree::
      :maxdepth: 1

      summary-exercices/answers_image_processing.rst
