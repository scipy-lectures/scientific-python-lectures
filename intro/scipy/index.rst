.. for doctests
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

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
:mod:`scipy.fft`         Fourier transform
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

"Special" functions are functions commonly used in science and mathematics that
are not considered to be "elementary" functions. Examples include

 * the gamma function, :func:`scipy.special.gamma`,
 * the error function, :func:`scipy.special.erf`,
 * Bessel functions, such as :func:`scipy.special.jv`
   (Bessel functions of the first kind), and
 * elliptic functions, such as :func:`scipy.special.ellipj`
   (Jacobian elliptic functions).

Other special functions are combinations of familiar elementary functions,
but they offer better accuracy or robustness than their naive implementations
would.

Most of these function are computed elementwise and follow standard
NumPy broadcasting rules when the input arrays have different shapes.
For example, :func:`scipy.special.xlog1py` is mathematically equivalent
to :math:`x\log(1 + y)`.

    >>> import scipy as sp
    >>> x = np.asarray([1, 2])
    >>> y = np.asarray([[3], [4], [5]])
    >>> res = sp.special.xlog1py(x, y)
    >>> res.shape
    (3, 2)
    >>> ref = x * np.log(1 + y)
    >>> np.allclose(res, ref)
    True

However, it is numerically favorable for small :math:`y`, when explicit addition
of ``1`` would lead to loss of precision due to floating point truncation error.

    >>> x = 2.5
    >>> y = 1e-18
    >>> x * np.log(1 + y)
    0.0
    >>> sp.special.xlog1py(x, y)
    2.5e-18

Many special functions also have "logarithmized" variants. For instance,
the gamma function :math:`\Gamma(\cdot)` is related to the factorial
function by :math:`n! = \Gamma(n + 1)`, but it extends the domain from the
positive integers to the complex plane.

   >>> x = np.arange(10)
   >>> np.allclose(sp.special.gamma(x + 1), sp.special.factorial(x))
   True
   >>> sp.special.gamma(5) < sp.special.gamma(5.5) < sp.special.gamma(6)
   True

The factorial function grows quickly, and so the gamma function overflows
for moderate values of the argument. However, sometimes only the logarithm
of the gamma function is needed. In such cases, we can compute the logarithm
of the gamma function directly using :func:`scipy.special.gammaln`.

   >>> x = [5, 50, 500]
   >>> np.log(sp.special.gamma(x))
   array([  3.17805383, 144.56574395,          inf])
   >>> sp.special.gammaln(x)
   array([   3.17805383,  144.56574395, 2605.11585036])

Such functions can often be used when the intermediate components of a
calculation would overflow or underflow, but the final result would not.
For example, suppose we wish to compute the ratio
:math:`\Gamma(500)/\Gamma(499)`.

    >>> a = sp.special.gamma(500)
    >>> b = sp.special.gamma(499)
    >>> a, b
    (inf, inf)

Both the numerator and denominator overflow, so performing $a / b$ will
not return the result we seek. However, the magnitude of the result should
be moderate, so the use of logarithms comes to mind. Combining the identities
:math:`\log(a/b) = \log(a) - \log(b)` and :math:`\exp(\log(x)) = x`,
we get:

    >>> log_a = sp.special.gammaln(500)
    >>> log_b = sp.special.gammaln(499)
    >>> log_res = log_a - log_b
    >>> res = np.exp(log_res)
    >>> res
    499.0000000...

Similarly, suppose we wish to compute the difference
:math:`\log(\Gamma(500) - \Gamma(499))`. For this, we use
:func:`scipy.special.logsumexp`, which computes
:math:`\log(\exp(x) + \exp(y))` using a numerical trick that avoids overflow.

    >>> res = sp.special.logsumexp([log_a, log_b],
    ...                            b=[1, -1])  # weights the terms of the sum
    >>> res
    2605.113844343...

For more information about these and many other special functions, see
the documentation of :mod:`scipy.special`.

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

Linear systems with special structure can often be solved more efficiently
than more general systems. For example, systems with triangular matrices
can be solved using :func:`scipy.linalg.solve_triangular`::

    >>> A_upper = np.triu(A)
    >>> A_upper
    array([[1, 2],
           [0, 3]])
    >>> np.allclose(sp.linalg.solve_triangular(A_upper, b, lower=False),
    ...             sp.linalg.solve(A_upper, b))
    True

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
    >>> rng = np.random.default_rng()
    >>> noise = (rng.random(10)*2 - 1) * 1e-1
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
               flag: converged
     function_calls: 12
         iterations: 6
               root: 1.0
             method: newton

.. warning::

    None of the functions in :mod:`scipy.optimize` that accept a guess are
    guaranteed to converge for all possible guesses! (For example, try
    ``x0=1.5`` in the example above, where the derivative of the function is
    exactly zero.) If this occurs, try a different guess, adjust the options
    (like providing a ``bracket`` as shown below), or consider whether SciPy
    offers a more appropriate method for the problem.

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

.. image:: auto_examples/images/sphx_glr_plot_optimize_example1_001.png
   :target: auto_examples/plot_optimize_example1.html
   :align: right
   :scale: 50

Suppose we wish to minimize the scalar-valued function of a single
variable :math:`f(x) = x^2  + 10 \sin(x)`::

    >>> def f(x):
    ...     return x**2 + 10*np.sin(x)
    >>> x = np.arange(-5, 5, 0.1)
    >>> plt.plot(x, f(x))
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.show()

We can see that the function has a local minimizer near :math:`x = 3.8`
and a global minimizer near :math:`x = -1.3`, but
the precise values cannot be determined from the plot.

The most appropriate function for this purpose is
:func:`scipy.optimize.minimize_scalar`.
Since we know the approximate locations of the minima, we will provide
bounds that restrict the search to the vicinity of the global minimum.

    >>> res = sp.optimize.minimize_scalar(f, bounds=(-2, -1))
    >>> res
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
    >>> res
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


:mod:`scipy.stats` contains fundamental tools for statistics in Python.

Statistical Distributions
.........................

Consider a random variable distributed according to the standard normal.
We draw a sample consisting of 100000 observations from the random variable.
The normalized histogram of the sample is an estimator of the random
variable's probability density function (PDF)::

    >>> dist = sp.stats.norm(loc=0, scale=1)  # standard normal distribution
    >>> sample = dist.rvs(size=100000)  # "random variate sample"
    >>> plt.hist(sample, bins=50, density=True, label='normalized histogram')  # doctest: +SKIP
    >>> x = np.linspace(-5, 5)
    >>> plt.plot(x, dist.pdf(x), label='PDF')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.legend()
    <matplotlib.legend.Legend object at ...>

.. image:: auto_examples/images/sphx_glr_plot_normal_distribution_001.png
    :target: auto_examples/plot_normal_distribution.html
    :scale: 70

.. sidebar:: **Distribution objects and frozen distributions**

   Each of the 100+ :mod:`scipy.stats` distribution families is represented by an
   *object* with a `__call__` method. Here, we call the :class:`scipy.stats.norm`
   object to specify its location and scale, and it returns a *frozen*
   distribution: a particular element of a distribution family with all
   parameters fixed. The frozen distribution object has methods to compute
   essential functions of the particular distribution.

Suppose we knew that the sample had been drawn from a distribution belonging
to the family of normal distributions, but we did not know the particular
distribution's location (mean) and scale (standard deviation). We perform
maximum likelihood estimation of the unknown parameters using the
distribution family's ``fit`` method::

    >>> loc, scale = sp.stats.norm.fit(sample)
    >>> loc
    0.0015767005...
    >>> scale
    0.9973396878...

Since we know the true parameters of the distribution from which the
sample was drawn, we are not surprised that these estimates are similar.

.. topic:: Exercise: Probability distributions
   :class: green

   Generate 1000 random variates from a gamma distribution with a shape
   parameter of 1. *Hint: the shape parameter is passed as the first
   argument when freezing the distribution*. Plot the histogram of the
   sample, and overlay the distribution's PDF. Estimate the shape parameter
   from the sample using the ``fit`` method.

   Extra: the distributions have many useful methods. Explore them
   using tab completion. Plot the cumulative density function of the
   distribution, and compute the variance.

Sample Statistics and Hypothesis Tests
......................................

The sample mean is an estimator of the mean of the distribution from which
the sample was drawn::

    >>> np.mean(sample)
    0.001576700508...

NumPy includes some of the most fundamental sample statistics (e.g.
:func:`numpy.mean`, :func:`numpy.var`, :func:`numpy.percentile`);
:mod:`scipy.stats` includes many more. For instance, the geometric mean
is a common measure of central tendency for data that tends to be
distributed over many orders of magnitude.

    >>> sp.stats.gmean(2**sample)
    1.0010934829...

SciPy also includes a variety of hypothesis tests that produce a
sample statistic and a p-value. For instance, suppose we wish to
test the null hypothesis that ``sample`` was drawn from a normal
distribution::

    >>> res = sp.stats.normaltest(sample)
    >>> res.statistic
    5.20841759...
    >>> res.pvalue
    0.07396163283...

Here, ``statistic`` is a sample statistic that tends to be high for
samples that are drawn from non-normal distributions. ``pvalue`` is
the probability of observing such a high value of the statistic for
a sample that *has* been drawn from a normal distribution. If the
p-value is unusually small, this may be taken as evidence that
``sample`` was *not* drawn from the normal distribution. Our statistic
and p-value are moderate, so the test is inconclusive.

There are many other features of :mod:`scipy.stats`, including circular
statistics, quasi-Monte Carlo methods, and resampling methods.
For much more information, see the documentation of :mod:`scipy.stats`
and the advanced chapter :ref:`statistics <statistics>`.

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
For example, :func:`scipy.integrate.solve_ivp` integrates ODEs of the form:

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

    >>> plt.plot(res.t, res.y[0])
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('t')
    Text(0.5, ..., 't')
    >>> plt.ylabel('y')
    Text(..., 0.5, 'y')
    >>> plt.title('Solution of Initial Value Problem')
    Text(0.5, 1.0, 'Solution of Initial Value Problem')

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

    \frac{dy}{dt} = \dot{y}
    \frac{d\dot{y}}{dt} = \ddot{y} = -(2 \zeta \omega_0  \dot{y} + \omega_0^2 y)

If we define :math:`z = [z_0, z_1]` where :math:`z_0 = y` and :math:`z_1 = \dot{y}`,
then the first order equation:

.. math::

    \frac{dz}{dt} =
    \begin{bmatrix}
        \frac{dz_0}{dt} \\
        \frac{dz_1}{dt}
    \end{bmatrix} =
    \begin{bmatrix}
        z_1  \\
        -(2 \zeta \omega_0  z_1 + \omega_0^2 z_0)
    \end{bmatrix}

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

Fast Fourier transforms: :mod:`scipy.fft`
---------------------------------------------

The :mod:`scipy.fft` module computes fast Fourier transforms (FFTs)
and offers utilities to handle them. Some important functions are:

* :func:`scipy.fft.fft` to compute the FFT

* :func:`scipy.fft.fftfreq` to generate the sampling frequencies

* :func:`scipy.fft.ifft` to compute the inverse FFT, from frequency
  space to signal space

|

As an illustration, a (noisy) input signal (``sig``), and its FFT::

    >>> sig_fft = sp.fft.fft(sig)  # doctest:+SKIP
    >>> freqs = sp.fft.fftfreq(sig.size, d=time_step)  # doctest:+SKIP


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

As the signal comes from a real-valued function, the Fourier transform is
symmetric.

The peak signal frequency can be found with ``freqs[power.argmax()]``

.. image:: auto_examples/images/sphx_glr_plot_fftpack_003.png
    :target: auto_examples/plot_fftpack.html
    :scale: 60
    :align: right


Setting the Fourier component above this frequency to zero and inverting
the FFT with :func:`scipy.fft.ifft`, gives a filtered signal.

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

   3. Find and use the 2-D FFT function in :mod:`scipy.fft`, and plot the
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

  >>> plt.plot(t, x)
  [<matplotlib.lines.Line2D object at ...>]
  >>> plt.plot(t[::4], x_resampled, 'ko')
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
  >>> rng = np.random.default_rng()
  >>> x = t + rng.normal(size=100)

  >>> x_detrended = sp.signal.detrend(x)

  >>> plt.plot(t, x)
  [<matplotlib.lines.Line2D object at ...>]
  >>> plt.plot(t, x_detrended)
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

.. only:: html

   **Exercises:**

.. toctree::
    :maxdepth: 1

    summary-exercises/stats-interpolate.rst
    summary-exercises/optimize-fit.rst
    summary-exercises/image-processing.rst

.. only:: html

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
