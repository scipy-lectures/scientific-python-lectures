.. for doctests
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")
    >>> import numpy as np
    >>> np.random.seed(0)

.. _scipy:

SciPy : high-level scientific computing
=======================================

**Authors**: *Gaël Varoquaux, Adrien Chauve, Andre Espaze, Emmanuelle Gouillart, Ralf Gommers*


.. topic:: Scipy

    The :mod:`scipy` package contains various toolboxes dedicated to common
    issues in scientific computing. Its different submodules correspond
    to different applications, such as interpolation, integration,
    optimization, image processing, statistics, special functions, etc.

.. tip::

    :mod:`scipy` can be compared to other standard scientific-computing
    libraries, such as the GSL (GNU Scientific  Library for C and C++),
    or Matlab's toolboxes. ``scipy`` is the core package for scientific
    routines in Python; it is meant to operate efficiently on ``numpy``
    arrays, so that NumPy and SciPy work hand in hand.

    Before implementing a routine, it is worth checking if the desired
    data processing is not already implemented in SciPy. As
    non-professional programmers, scientists often tend to **re-invent the
    wheel**, which leads to buggy, non-optimal, difficult-to-share and
    unmaintainable code. By contrast, ``SciPy``'s routines are optimized
    and tested, and should therefore be used when possible.


.. contents:: Chapters contents
    :local:
    :depth: 1


.. warning::

    This tutorial is far from an introduction to numerical computing.
    As enumerating the different submodules and functions in SciPy would
    be very boring, we concentrate instead on a few examples to give a
    general idea of how to use ``scipy`` for scientific computing.

:mod:`scipy` is composed of task-specific sub-modules:

=========================== ==========================================
:mod:`scipy.cluster`         Vector quantization / Kmeans
:mod:`scipy.constants`       Physical and mathematical constants
:mod:`scipy.fftpack`         Fourier transform
:mod:`scipy.integrate`       Integration routines
:mod:`scipy.interpolate`     Interpolation
:mod:`scipy.io`              Data input and output
:mod:`scipy.linalg`          Linear algebra routines
:mod:`scipy.ndimage`         n-dimensional image package
:mod:`scipy.odr`             Orthogonal distance regression
:mod:`scipy.optimize`        Optimization
:mod:`scipy.signal`          Signal processing
:mod:`scipy.sparse`          Sparse matrices
:mod:`scipy.spatial`         Spatial data structures and algorithms
:mod:`scipy.special`         Any special mathematical functions
:mod:`scipy.stats`           Statistics
=========================== ==========================================

.. tip::

   They all depend on :mod:`numpy`, but are mostly independent of each
   other. The standard way of importing NumPy and these SciPy modules
   is::

    >>> import numpy as np
    >>> import scipy as sp


File input/output: :mod:`scipy.io`
----------------------------------
:mod:`scipy.io` contains functions for loading and saving data in
several common formats including Matlab, IDL, Matrix Market, and
Harwell-Boeing.

**Matlab files**: Loading and saving::

    >>> import scipy as sp
    >>> a = np.ones((3, 3))
    >>> sp.io.savemat('file.mat', {'a': a})  # savemat expects a dictionary
    >>> data = sp.io.loadmat('file.mat')
    >>> data['a']
    array([[1.,  1.,  1.],
           [1.,  1.,  1.],
           [1.,  1.,  1.]])

.. warning:: **Python / Matlab mismatch**: The Matlab file format does not support 1D arrays.

   ::

      >>> a = np.ones(3)
      >>> a
      array([1.,  1.,  1.])
      >>> a.shape
      (3,)
      >>> sp.io.savemat('file.mat', {'a': a})
      >>> a2 = sp.io.loadmat('file.mat')['a']
      >>> a2
      array([[1.,  1.,  1.]])
      >>> a2.shape
      (1, 3)

   Notice that the original array was a one-dimensional array, whereas the
   saved and reloaded array is a two-dimensional array with a single row.

   For other formats, see the :mod:`scipy.io` documentation.

.. seealso::

    * Load text files: :func:`numpy.loadtxt`/:func:`numpy.savetxt`

    * Clever loading of text/csv files:
      :func:`numpy.genfromtxt`

    * Fast and efficient, but NumPy-specific, binary format:
      :func:`numpy.save`/:func:`numpy.load`

    * Basic input/output of images in Matplotlib:
      :func:`matplotlib.pyplot.imread`/:func:`matplotlib.pyplot.imsave`

    * More advanced input/output of images: :mod:`imageio`

Special functions: :mod:`scipy.special`
---------------------------------------

Special functions are transcendental functions. The docstring of the
:mod:`scipy.special` module is well-written, so we won't list all
functions here. Frequently used ones are:

 * Bessel function, such as :func:`scipy.special.jn` (nth integer order Bessel
   function)

 * Elliptic function (:func:`scipy.special.ellipj` for the Jacobian elliptic
   function, ...)

 * Gamma function: :func:`scipy.special.gamma`, also note
   :func:`scipy.special.gammaln` which
   will give the log of Gamma to a higher numerical precision.

 * Erf, the area under a Gaussian curve: :func:`scipy.special.erf`


.. _scipy_linalg:

Linear algebra operations: :mod:`scipy.linalg`
----------------------------------------------

:mod:`scipy.linalg` provides a Python interface to efficient, compiled
implementations of standard linear algebra operations: the BLAS (Basic
Linear Algebra Subroutines) and LAPACK (Linear Algebra PACKage) libraries.

For example, the :func:`scipy.linalg.det` function computes the determinant
of a square matrix::

    >>> import scipy as sp
    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> sp.linalg.det(arr)
    -2.0
    >>> arr = np.array([[1, 1],
    ...                 [1, 1]])
    >>> sp.linalg.det(arr)
    0.0
    >>> sp.linalg.det(np.ones((3, 4)))
    Traceback (most recent call last):
    ...
    ValueError: expected square matrix

Mathematically, the solution of a linear system :math:`Ax = b` is :math:`x = A^{-1}b`,
but explicit inversion of a matrix is numerically unstable and should be avoided.
Instead, use :func:`scipy.linalg.solve`::

    >>> A = np.array([[1, 2],
    ...               [2, 3]])
    >>> b = np.array([14, 23])
    >>> x = sp.linalg.solve(A, b)
    >>> x
    array([4., 5.])
    >>> np.allclose(A @ x, b)
    True

Attempting to solve a linear system involving a singular matrix
(i.e. one with a determinant of zero) will raise a ``LinAlgError``::

    >>> A = np.array([[1, 1],
    ...               [1, 1]])
    >>> sp.linalg.solve(A, b)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    ...LinAlgError: Matrix is singular

:mod:`scipy.linalg` also features matrix factorizations/decompositions
such as the singular value decomposition.

    >>> A = np.array([[1, 2],
    ...               [2, 3]])
    >>> U, s, Vh = sp.linalg.svd(A)
    >>> s  # singular values
    array([4.23606798, 0.23606798])

The original matrix can be recovered by matrix multiplication of the
factors::

    >>> S = np.diag(s)  # convert to diagonal matrix before matrix multiplication
    >>> A2 = U @ S @ Vh
    >>> np.allclose(A2, A)
    True
    >>> A3 = (U * s) @ Vh  # more efficient: use array math broadcasting rules!
    >>> np.allclose(A3, A)
    True

Many other decompositions (e.g. LU, Cholesky, QR), solvers for structured
linear systems (e.g. triangular, circulant), eigenvalue problem algorithms,
matrix functions (e.g. matrix exponential), and routines for special matrix
creation (e.g. block diagonal, toeplitz) are available in :mod:`scipy.linalg`.


.. _intro_scipy_interpolate:

Interpolation: :mod:`scipy.interpolate`
---------------------------------------

:mod:`scipy.interpolate` is useful for fitting a function from experimental
data and thus evaluating points where no reference value exists. The module
includes, but not limited to `FITPACK Fortran subroutines`_.

.. _`FITPACK Fortran subroutines` : https://netlib.org/dierckx/index.html
.. _netlib : https://netlib.org

By imagining experimental data close to a sine function::

    >>> measured_time = np.linspace(0, 1, 10)
    >>> noise = (np.random.random(10)*2 - 1) * 1e-1
    >>> measures = np.sin(2 * np.pi * measured_time) + noise

:mod:`scipy.interpolate` has many interpolation methods which need to be
chosen based on the data. See the
`tutorial <https://scipy.github.io/devdocs/tutorial/interpolate.html>`__
for some guidelines::

    >>> spline = sp.interpolate.CubicSpline(measured_time, measures)

.. image:: auto_examples/images/sphx_glr_plot_interpolation_001.png
    :target: auto_examples/plot_interpolation.html
    :scale: 60
    :align: right

Then the result can be evaluated at the time of interest::

    >>> interpolation_time = np.linspace(0, 1, 50)
    >>> linear_results = spline(interpolation_time)

:class:`scipy.interpolate.CloughTocher2DInterpolator` is similar to
:class:`scipy.interpolate.CubicSpline`, but for 2-D arrays.
See the summary exercise on
:ref:`summary_exercise_stat_interp` for a more advanced spline
interpolation example.



Optimization and fit: :mod:`scipy.optimize`
-------------------------------------------

:mod:`scipy.optimize` provides algorithms for root finding, curve fitting,
and more general optimization.

Root Finding
............

:func:`scipy.optimize.root_scalar` attempts to find a root of a specified
scalar-valued function (i.e., an argument at which the function value is zero).
Like many :mod:`scipy.optimize` functions, the function needs an initial
guess of the solution, which the algorithm will refine until it converges or
recognizes failure. We also provide the derivative to improve the rate of
convergence.

    >>> def f(x):
    ...     return (x-1)*(x-2)
    >>> def df(x):
    ...     return 2*x - 3
    >>> x0 = 0  # guess
    >>> res = sp.optimize.root_scalar(f, x0=x0, fprime=df)
    >>> res
         converged: True
              flag: 'converged'
    function_calls: 12
        iterations: 6
              root: 1.0

Note that only one the root at ``1.0`` is found. By inspection, we can tell
that there is a second root at ``2.0``. We can direct the function toward a
particular root by changing the guess or by passing a bracket that contains
only the root we seek.

    >>> res = sp.optimize.root_scalar(f, bracket=(1.5, 10))
    >>> res.root
    2.0

For multivariate problems, use :func:`scipy.optimize.root`.

    >>> def f(x):
    ...     # intersection of unit circle and line from origin
    ...     return [x[0]**2 + x[1]**2 - 1,
    ...             x[1] - x[0]]
    >>> res = sp.optimize.root(f, x0=[0, 0])
    >>> np.allclose(f(res.x), 0, atol=1e-10)
    True
    >>> np.allclose(res.x, np.sqrt(2)/2)
    True

Over-constrained problems can be solved in the least-squares
sense using :func:`scipy.optimize.root` with ``method='lm'``
(Levenberg-Marquardt).

    >>> def f(x):
    ...     # intersection of unit circle, line from origin, and parabola
    ...     return [x[0]**2 + x[1]**2 - 1,
    ...             x[1] - x[0],
    ...             x[1] - x[0]**2]
    >>> res = sp.optimize.root(f, x0=[1, 1], method='lm')
    >>> res.success
    True
    >>> res.x
    array([0.76096066, 0.66017736])

See the documentation of :func:`scipy.optimize.root_scalar`
and :func:`scipy.optimize.root` for a variety of other solution
algorithms and options.

Curve fitting
.............

.. Comment to make doctest pass
    >>> np.random.seed(0)

.. image:: auto_examples/images/sphx_glr_plot_curve_fit_001.png
   :target: auto_examples/plot_curve_fit.html
   :align: right
   :scale: 50

Suppose we have data that is sinusoidal but noisy::

    >>> x = np.linspace(-5, 5, num=50)  # 50 values between -5 and 5
    >>> noise = 0.01 * np.cos(100 * x)
    >>> a, b = 2.9, 1.5
    >>> y = a * np.cos(b * x) + noise

We can approximate the underlying amplitude, frequency, and phase
from the data by least squares curve fitting. To begin, we write
a function that accepts the independent variable as the first
argument and all parameters to fit as separate arguments::

    >>> def f(x, a, b, c):
    ...     return a * np.sin(b * x + c)

.. image:: auto_examples/images/sphx_glr_plot_curve_fit_002.png
   :target: auto_examples/plot_curve_fit.html
   :align: right
   :scale: 50

We then use :func:`scipy.optimize.curve_fit` to find :math:`a` and :math:`b`::

    >>> params, _ = sp.optimize.curve_fit(f, x, y, p0=[2, 1, 3])
    >>> params
    array([2.900026  , 1.50012043, 1.57079633])
    >>> ref = [a, b, np.pi/2]  # what we'd expect
    >>> np.allclose(params, ref, rtol=1e-3)
    True

.. raw:: html

   <div style="clear: both"></div>

.. topic:: Exercise: Curve fitting of temperature data
   :class: green

    The temperature extremes in Alaska for each month, starting in January, are
    given by (in degrees Celsius)::

        max:  17,  19,  21,  28,  33,  38, 37,  37,  31,  23,  19,  18
        min: -62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58

    1. Plot these temperature extremes.
    2. Define a function that can describe min and max temperatures.
       Hint: this function has to have a period of 1 year.
       Hint: include a time offset.
    3. Fit this function to the data with :func:`scipy.optimize.curve_fit`.
    4. Plot the result.  Is the fit reasonable?  If not, why?
    5. Is the time offset for min and max temperatures the same within the fit
       accuracy?

    :ref:`solution <sphx_glr_intro_scipy_auto_examples_solutions_plot_curvefit_temperature_data.py>`


Optimization
............

.. Comment to make doctest pass
    >>> np.random.seed(0)

.. image:: auto_examples/images/sphx_glr_plot_optimize_example1_001.png
   :target: auto_examples/plot_optimize_example1.html
   :align: right
   :scale: 50

Suppose we wish to minimize the scalar-valued function of a single
variable :math:`f(x) = x^2  + 10 \sin(x)`::

    >>> def f(x):
    ...     return x**2 + 10*np.sin(x)
    >>> x = np.arange(-5, 5, 0.1)
    >>> plt.plot(x, f(x))  # doctest:+SKIP
    >>> plt.show()  # doctest:+SKIP

We can see that the function has a local minimizer near :math:`x = 3.8`
and a global minimizer near :math:`x = -1.3`, but
the precise values cannot be determined from the plot.

The most appropriate function for this purpose is
:func:`scipy.optimize.minimize_scalar`.
Since we know the approximate locations of the minima, we will provide
bounds that restrict the search to the vicinity of the global minimum.

    >>> res = sp.optimize.minimize_scalar(f, bounds=(-2, -1))
    >>> res  # doctest: +ELLIPSIS
     message: Solution found.
     success: True
      status: 0
         fun: -7.9458233756...
           x: -1.306440997...
         nit: 8
        nfev: 8
    >>> res.fun == f(res.x)
    True

If we did not already know the approximate location of the global minimum,
we could use one of SciPy's global minimizers, such as
:func:`scipy.optimize.differential_evolution`. We are required to pass
``bounds``, but they do not need to be tight.

    >>> bounds=[(-5, 5)]  # list of lower, upper bound for each variable
    >>> res = sp.optimize.differential_evolution(f, bounds=bounds)
    >>> res  # doctest:+SKIP
     message: Optimization terminated successfully.
     success: True
         fun: -7.9458233756...
           x: [-1.306e+00]
         nit: 6
        nfev: 111
         jac: [ 9.948e-06]

For multivariate optimization, a good choice for many problems is
:func:`scipy.optimize.minimize`.
Suppose we wish to find the minimum of a quadratic function of two
variables, :math:`f(x_0, x_1) = (x_0-1)^2 + (x_1-2)^2`.

    >>> def f(x):
    ...     return (x[0] - 1)**2 + (x[1] - 2)**2

Like :func:`scipy.optimize.root`, :func:`scipy.optimize.minimize`
requires a guess ``x0``. (Note that this is the initial value of
*both* variables rather than the value of the variable we happened to
label :math:`x_0`.)

    >>> res = sp.optimize.minimize(f, x0=[0, 0])
    >>> res  # doctest: +ELLIPSIS
      message: Optimization terminated successfully.
      success: True
       status: 0
          fun: 1.70578...e-16
            x: [ 1.000e+00  2.000e+00]
          nit: 2
          jac: [ 3.219e-09 -8.462e-09]
     hess_inv: [[ 9.000e-01 -2.000e-01]
                [-2.000e-01  6.000e-01]]
         nfev: 9
         njev: 3

.. sidebar:: **Maximization?**

   Is :func:`scipy.optimize.minimize` restricted to the solution of
   minimization problems? Nope! To solve a maximization problem,
   simply minimize the *negative* of the original objective function.

This barely scratches the surface of SciPy's optimization features, which
include mixed integer linear programming, constrained nonlinear programming,
and the solution of assignment problems. For much more information, see the
documentation of :mod:`scipy.optimize` and the advanced chapter
:ref:`mathematical_optimization`.

.. topic:: Exercise: 2-D minimization
   :class: green

    .. image:: auto_examples/images/sphx_glr_plot_2d_minimization_002.png
        :target: auto_examples/plot_2d_minimization.html
        :align: right
        :scale: 50

    The six-hump camelback function

    .. math:: f(x, y) = (4 - 2.1x^2 + \frac{x^4}{3})x^2 + xy + (4y^2 - 4)y^2

    has multiple local minima. Find a global minimum (there is more than one,
    each with the same value of the objective function) and at least one other
    local minimum.

    Hints:

        - Variables can be restricted to :math:`-2 < x < 2` and :math:`-1 < y < 1`.
        - :func:`numpy.meshgrid` and :func:`matplotlib.pyplot.imshow` can help
          with visualization.
        - Try minimizing with :func:`scipy.optimize.minimize` with an initial
          guess of :math:`(x, y) = (0, 0)`. Does it find the global minimum, or
          converge to a local minimum? What about other initial guesses?
        - Try minimizing with :func:`scipy.optimize.differential_evolution`.

    :ref:`solution <sphx_glr_intro_scipy_auto_examples_plot_2d_minimization.py>`

See the summary exercise on :ref:`summary_exercise_optimize` for another, more
advanced example.


Statistics and random numbers: :mod:`scipy.stats`
-------------------------------------------------

.. Comment to make doctest pass
    >>> np.random.seed(0)


The module :mod:`scipy.stats` contains statistical tools and probabilistic
descriptions of random processes. Random number generators for various
random process can be found in :mod:`numpy.random`.

Distributions: histogram and probability density function
..........................................................

Given observations of a random process, their histogram is an estimator of
the random process's PDF (probability density function): ::

    >>> samples = np.random.normal(size=1000)
    >>> bins = np.arange(-4, 5)
    >>> bins
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> histogram = np.histogram(samples, bins=bins, density=True)[0]
    >>> bins = 0.5*(bins[1:] + bins[:-1])
    >>> bins
    array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5])
    >>> pdf = sp.stats.norm.pdf(bins)  # norm is a distribution object

    >>> plt.plot(bins, histogram) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(bins, pdf) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]

.. image:: auto_examples/images/sphx_glr_plot_normal_distribution_001.png
    :target: auto_examples/plot_normal_distribution.html
    :scale: 70

.. sidebar:: **The distribution objects**

   :class:`scipy.stats.norm` is a distribution object: each distribution
   in :mod:`scipy.stats` is represented as an object. Here it's the
   normal distribution, and it comes with a PDF, a CDF, and much more.

If we know that the random process belongs to a given family of random
processes, such as normal processes, we can do a maximum-likelihood fit
of the observations to estimate the parameters of the underlying
distribution. Here we fit a normal process to the observed data::

    >>> loc, std = sp.stats.norm.fit(samples)
    >>> loc     # doctest: +ELLIPSIS
    -0.045256707...
    >>> std     # doctest: +ELLIPSIS
    0.9870331586...

.. topic:: Exercise: Probability distributions
   :class: green

   Generate 1000 random variates from a gamma distribution with a shape
   parameter of 1, then plot a histogram from those samples.  Can you plot the
   pdf on top (it should match)?

   Extra: the distributions have many useful methods. Explore them by
   reading the docstring or by using tab completion.  Can you recover
   the shape parameter 1 by using the ``fit`` method on your random
   variates?


Mean, median and percentiles
.............................

The mean is an estimator of the center of the distribution::

    >>> np.mean(samples)     # doctest: +ELLIPSIS
    -0.0452567074...

The median another estimator of the center. It is the value with half of
the observations below, and half above::

    >>> np.median(samples)     # doctest: +ELLIPSIS
    -0.0580280347...

.. tip::

   Unlike the mean, the median is not sensitive to the tails of the
   distribution. It is `"robust"
   <https://en.wikipedia.org/wiki/Robust_statistics>`_.

.. topic:: Exercise: Compare mean and median on samples of a Gamma distribution
   :class: green

    Which one seems to be the best estimator of the center for the Gamma
    distribution?


The median is also the percentile 50, because 50% of the observation are
below it::

    >>> sp.stats.scoreatpercentile(samples, 50)     # doctest: +ELLIPSIS
    -0.0580280347...

Similarly, we can calculate the percentile 90::

    >>> sp.stats.scoreatpercentile(samples, 90)     # doctest: +ELLIPSIS
    1.2315935511...

.. tip::

    The percentile is an estimator of the CDF: cumulative distribution
    function.

Statistical tests
.................

.. image:: auto_examples/images/sphx_glr_plot_t_test_001.png
    :target: auto_examples/plot_t_test.html
    :scale: 60
    :align: right

A statistical test is a decision indicator. For instance, if we have two
sets of observations, that we assume are generated from Gaussian
processes, we can use a
`T-test <https://en.wikipedia.org/wiki/Student%27s_t-test>`__ to decide
whether the means of two sets of observations are significantly different::

    >>> a = np.random.normal(0, 1, size=100)
    >>> b = np.random.normal(1, 1, size=10)
    >>> sp.stats.ttest_ind(a, b)   # doctest: +SKIP
    (array(-3.177574054...), 0.0019370639...)

.. tip:: The resulting output is composed of:

    * The T statistic value: it is a number the sign of which is
      proportional to the difference between the two random processes and
      the magnitude is related to the significance of this difference.

    * the *p value*: the probability of both processes being identical. If it
      is close to 1, the two process are almost certainly identical.
      The closer it is to zero, the more likely it is that the processes
      have different means.

.. seealso::

   The chapter on :ref:`statistics <statistics>` introduces much more
   elaborate tools for statistical testing and statistical data
   loading and visualization outside of scipy.

Numerical integration: :mod:`scipy.integrate`
---------------------------------------------

Quadrature
..........

Suppose we wish to compute the definite integral
:math:`\int_0^{\pi / 2} \sin(t) dt` numerically. :func:`scipy.integrate.quad`
chooses one of several adaptive techniques depending on the parameters, and
is therefore the recommended first choice for integration of function of a single variable::

    >>> integral, error_estimate = sp.integrate.quad(np.sin, 0, np.pi/2)
    >>> np.allclose(integral, 1)  # numerical result ~ analytical result
    True
    >>> abs(integral - 1) < error_estimate  #  actual error < estimated error
    True

Other functions for *numerical quadrature*, including integration of
multivariate functions and approximating integrals from samples, are available
in :mod:`scipy.integrate`.

Initial Value Problems
......................

:mod:`scipy.integrate` also features routines for integrating `Ordinary
Differential Equations (ODE)
<https://en.wikipedia.org/wiki/Ordinary_differential_equation>`__.
For example, :func:`scipy.integrate.solve_ivp` integrates ODEs of the form::

.. math::

    \frac{dy}{dt} = f(t, y(t))

from an initial time :math:`t_0` and initial state :math:`y(t=t_0)=t_0` to a final
time :math:`t_f` or until an event occurs (e.g. a specified state is reached).

As an introduction, consider the initial value problem given by
:math:`\frac{dy}{dt} = -2 y` and the initial condition :math:`y(t=0) = 1` on
the interval :math:`t = 0 \dots 4`. We begin by defining a callable that
computes :math:`f(t, y(t))` given the current time and state.

    >>> def f(t, y):
    ...     return -2 * y

Then, to compute ``y`` as a function of time::

    >>> t_span = (0, 4)  # time interval
    >>> t_eval = np.linspace(*t_span)  # times at which to evaluate `y`
    >>> y0 = [1,]  # initial state
    >>> res = sp.integrate.solve_ivp(f, t_span=t_span, y0=y0, t_eval=t_eval)

and plot the result::

    >>> plt.plot(res.t, res.y[0])  # doctest: +SKIP
    >>> plt.xlabel('t')  # doctest: +SKIP
    >>> plt.ylabel('y')  # doctest: +SKIP
    >>> plt.title('Solution of Initial Value Problem')  # doctest: +SKIP

.. image:: auto_examples/images/sphx_glr_plot_solve_ivp_simple_001.png
    :target: auto_examples/plot_solve_ivp_simple.html
    :scale: 70
    :align: right

Let us integrate a more complex ODE: a `damped
spring-mass oscillator
<https://en.wikipedia.org/wiki/Harmonic_oscillator#Damped_harmonic_oscillator>`__.
The position of a mass attached to a spring obeys the 2nd order ODE
:math:`\ddot{y} + 2 \zeta \omega_0  \dot{y} + \omega_0^2 y = 0` with natural frequency
:math:`\omega_0 = \sqrt{k/m}`, damping ratio :math:`\zeta = c/(2 m \omega_0)`,
spring constant :math:`k`, mass :math:`m`, and damping coefficient :math:`c`.

Before using :func:`scipy.integrate.solve_ivp`, the 2nd order ODE
needs to be transformed into a system of first-order ODEs. Note that

.. math::

    \begin{eqnarray}
    \frac{dy}{dt} &=& \dot{y} \\
    \frac{d\dot{y}}{dt} &=& \ddot{y} = -(2 \zeta \omega_0  \dot{y} + \omega_0^2 y)
    \end{eqnarray}

If we define :math:`z = [z_0, z_1]` where :math:`z_0 = y` and :math:`z_1 = \dot{y}`,
then the first order equation:

.. math::

    \usepackage{amsmath}
    \begin{eqnarray}
    \frac{dz}{dt} =
    \begin{bmatrix}
        \frac{dz_0}{dt} \\
        \frac{dz_1}{dt}
    \end{bmatrix} =
    \begin{bmatrix}
        z_1  \\
        -(2 \zeta \omega_0  z_1 + \omega_0^2 z_0)
    \end{bmatrix}
    \end{eqnarray}

is equivalent to the original second order equation.

We set::

    >>> m = 0.5  # kg
    >>> k = 4  # N/m
    >>> c = 0.4  # N s/m
    >>> zeta = c / (2 * m * np.sqrt(k/m))
    >>> omega = np.sqrt(k / m)

and define the function that computes :math:`\dot{z} = f(t, z(t))`::

    >>> def f(t, z, zeta, omega):
    ...     return (z[1], -2.0 * zeta * omega * z[1] - omega**2 * z[0])

.. image:: auto_examples/images/sphx_glr_plot_solve_ivp_damped_spring_mass_001.png
    :target: auto_examples/plot_solve_ivp_damped_spring_mass.html
    :scale: 70
    :align: right

Integration of the system follows::

    >>> t_span = (0, 10)
    >>> t_eval = np.linspace(*t_span, 100)
    >>> z0 = [1, 0]
    >>> res = sp.integrate.solve_ivp(f, t_span, z0, t_eval=t_eval,
    ...                              args=(zeta, omega), method='LSODA')

.. tip::

    With the option `method='LSODA'`, :func:`scipy.integrate.solve_ivp` uses the LSODA
    (Livermore Solver for Ordinary Differential equations with Automatic method switching
    for stiff and non-stiff problems). See the `ODEPACK Fortran library`_ for more details.

.. _`ODEPACK Fortran library` : https://people.sc.fsu.edu/~jburkardt/f77_src/odepack/odepack.html

.. seealso:: **Partial Differental Equations**

    There is no Partial Differential Equations (PDE) solver in SciPy.
    Some Python packages for solving PDE's are available, such as fipy_
    or SfePy_.

.. _fipy: https://www.ctcms.nist.gov/fipy/
.. _SfePy: https://sfepy.org/doc/

Fast Fourier transforms: :mod:`scipy.fftpack`
---------------------------------------------

The :mod:`scipy.fftpack` module computes fast Fourier transforms (FFTs)
and offers utilities to handle them. The main functions are:

* :func:`scipy.fftpack.fft` to compute the FFT

* :func:`scipy.fftpack.fftfreq` to generate the sampling frequencies

* :func:`scipy.fftpack.ifft` computes the inverse FFT, from frequency
  space to signal space

|

As an illustration, a (noisy) input signal (``sig``), and its FFT::

    >>> sig_fft = sp.fftpack.fft(sig) # doctest:+SKIP
    >>> freqs = sp.fftpack.fftfreq(sig.size, d=time_step) # doctest:+SKIP


.. |signal_fig| image:: auto_examples/images/sphx_glr_plot_fftpack_001.png
    :target: auto_examples/plot_fftpack.html
    :scale: 60

.. |fft_fig| image:: auto_examples/images/sphx_glr_plot_fftpack_002.png
    :target: auto_examples/plot_fftpack.html
    :scale: 60

===================== =====================
|signal_fig|          |fft_fig|
===================== =====================
**Signal**            **FFT**
===================== =====================

As the signal comes from a real function, the Fourier transform is
symmetric.

The peak signal frequency can be found with ``freqs[power.argmax()]``

.. image:: auto_examples/images/sphx_glr_plot_fftpack_003.png
    :target: auto_examples/plot_fftpack.html
    :scale: 60
    :align: right


Setting the Fourrier component above this frequency to zero and inverting
the FFT with :func:`scipy.fftpack.ifft`, gives a filtered signal.

.. note::

   The code of this example can be found :ref:`here <sphx_glr_intro_scipy_auto_examples_plot_fftpack.py>`

.. topic:: `numpy.fft`

   NumPy also has an implementation of FFT (:mod:`numpy.fft`). However,
   the SciPy one
   should be preferred, as it uses more efficient underlying implementations.

|

**Fully worked examples:**

.. |periodicity_finding| image:: auto_examples/solutions/images/sphx_glr_plot_periodicity_finder_001.png
    :scale: 50
    :target: auto_examples/solutions/plot_periodicity_finder.html

.. |image_blur| image:: auto_examples/solutions/images/sphx_glr_plot_image_blur_002.png
    :scale: 50
    :target: auto_examples/solutions/plot_image_blur.html

=================================================================================================================== ===================================================================================================================
Crude periodicity finding (:ref:`link <sphx_glr_intro_scipy_auto_examples_solutions_plot_periodicity_finder.py>`)   Gaussian image blur (:ref:`link <sphx_glr_intro_scipy_auto_examples_solutions_plot_image_blur.py>`)
=================================================================================================================== ===================================================================================================================
|periodicity_finding|                                                                                               |image_blur|
=================================================================================================================== ===================================================================================================================

|

.. topic:: Exercise: Denoise moon landing image
   :class: green

   .. image:: ../../data/moonlanding.png
     :scale: 70

   1. Examine the provided image :download:`moonlanding.png
      <../../data/moonlanding.png>`, which is heavily contaminated with periodic
      noise. In this exercise, we aim to clean up the noise using the
      Fast Fourier Transform.

   2. Load the image using :func:`matplotlib.pyplot.imread`.

   3. Find and use the 2-D FFT function in :mod:`scipy.fftpack`, and plot the
      spectrum (Fourier transform of) the image. Do you have any trouble
      visualising the spectrum? If so, why?

   4. The spectrum consists of high and low frequency components. The noise is
      contained in the high-frequency part of the spectrum, so set some of
      those components to zero (use array slicing).

   5. Apply the inverse Fourier transform to see the resulting image.

   :ref:`Solution <sphx_glr_intro_scipy_auto_examples_solutions_plot_fft_image_denoise.py>`

|


Signal processing: :mod:`scipy.signal`
--------------------------------------

.. tip::

   :mod:`scipy.signal` is for typical signal processing: 1D,
   regularly-sampled signals.

.. image:: auto_examples/images/sphx_glr_plot_resample_001.png
    :target: auto_examples/plot_resample.html
    :scale: 65
    :align: right


**Resampling** :func:`scipy.signal.resample`: resample a signal to `n`
points using FFT. ::

  >>> t = np.linspace(0, 5, 100)
  >>> x = np.sin(t)

  >>> x_resampled = sp.signal.resample(x, 25)

  >>> plt.plot(t, x) # doctest: +ELLIPSIS
  [<matplotlib.lines.Line2D object at ...>]
  >>> plt.plot(t[::4], x_resampled, 'ko') # doctest: +ELLIPSIS
  [<matplotlib.lines.Line2D object at ...>]

.. tip::

    Notice how on the side of the window the resampling is less accurate
    and has a rippling effect.

    This resampling is different from the :ref:`interpolation
    <intro_scipy_interpolate>` provided by :mod:`scipy.interpolate` as it
    only applies to regularly sampled data.


.. image:: auto_examples/images/sphx_glr_plot_detrend_001.png
    :target: auto_examples/plot_detrend.html
    :scale: 65
    :align: right

**Detrending** :func:`scipy.signal.detrend`: remove linear trend from signal::

  >>> t = np.linspace(0, 5, 100)
  >>> x = t + np.random.normal(size=100)

  >>> x_detrended = sp.signal.detrend(x)

  >>> plt.plot(t, x) # doctest: +ELLIPSIS
  [<matplotlib.lines.Line2D object at ...>]
  >>> plt.plot(t, x_detrended) # doctest: +ELLIPSIS
  [<matplotlib.lines.Line2D object at ...>]

.. raw:: html

   <div style="clear: both"></div>

**Filtering**:
For non-linear filtering, :mod:`scipy.signal` has filtering (median
filter :func:`scipy.signal.medfilt`, Wiener :func:`scipy.signal.wiener`),
but we will discuss this in the image section.

.. tip::

    :mod:`scipy.signal` also has a full-blown set of tools for the design
    of linear filter (finite and infinite response filters), but this is
    out of the scope of this tutorial.


**Spectral analysis**:
:func:`scipy.signal.spectrogram` compute a spectrogram --frequency
spectrums over consecutive time windows--, while
:func:`scipy.signal.welch` comptes a power spectrum density (PSD).

.. |chirp_fig| image:: auto_examples/images/sphx_glr_plot_spectrogram_001.png
    :target: auto_examples/plot_spectrogram.html
    :scale: 45

.. |spectrogram_fig| image:: auto_examples/images/sphx_glr_plot_spectrogram_002.png
    :target: auto_examples/plot_spectrogram.html
    :scale: 45

.. |psd_fig| image:: auto_examples/images/sphx_glr_plot_spectrogram_003.png
    :target: auto_examples/plot_spectrogram.html
    :scale: 45

|chirp_fig| |spectrogram_fig| |psd_fig|

Image manipulation: :mod:`scipy.ndimage`
-----------------------------------------

.. include:: image_processing/image_processing.rst
    :start-line: 1


Summary exercises on scientific computing
-----------------------------------------

The summary exercises use mainly NumPy, SciPy and Matplotlib. They provide some
real-life examples of scientific computing with Python. Now that the basics of
working with NumPy and SciPy have been introduced, the interested user is
invited to try these exercises.

.. only:: latex

    .. toctree::
       :maxdepth: 1

       summary-exercises/stats-interpolate.rst
       summary-exercises/optimize-fit.rst
       summary-exercises/image-processing.rst
       summary-exercises/answers_image_processing.rst

.. only:: html

   **Exercises:**

   .. toctree::
       :maxdepth: 1

       summary-exercises/stats-interpolate.rst
       summary-exercises/optimize-fit.rst
       summary-exercises/image-processing.rst

   **Proposed solutions:**

   .. toctree::
      :maxdepth: 1

      summary-exercises/answers_image_processing.rst

.. include the gallery. Skip the first line to avoid the "orphan"
   declaration

.. include:: auto_examples/index.rst
    :start-line: 1


.. seealso:: **References to go further**

   * Some chapters of the `advanced <advanced_topics_part>`__ and the
     `packages and applications <applications_part>`__ parts of the SciPy
     lectures

   * The `SciPy cookbook <https://scipy-cookbook.readthedocs.io>`__

.. compile solutions, but don't list them explicitly
.. toctree::
   :hidden:

   solutions.rst
