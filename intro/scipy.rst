.. for doctests
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")
    >>> import numpy as np
    >>> np.random.seed(0)

.. _scipy:

Scipy : high-level scientific computing
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
    arrays, so that numpy and scipy work hand in hand.

    Before implementing a routine, it is worth checking if the desired
    data processing is not already implemented in Scipy. As
    non-professional programmers, scientists often tend to **re-invent the
    wheel**, which leads to buggy, non-optimal, difficult-to-share and
    unmaintainable code. By contrast, ``Scipy``'s routines are optimized
    and tested, and should therefore be used when possible.


.. contents:: Chapters contents
    :local:
    :depth: 1


.. warning::

    This tutorial is far from an introduction to numerical computing.
    As enumerating the different submodules and functions in scipy would
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
   other. The standard way of importing Numpy and these Scipy modules
   is::

    >>> import numpy as np
    >>> from scipy import stats  # same for other sub-modules

   The main :mod:`scipy` namespace mostly contains functions that are really
   numpy functions (try ``scipy.cos is np.cos``). Those are exposed for
   historical reasons; there's no reason to use ``import
   scipy`` in your code.


File input/output: :mod:`scipy.io`
----------------------------------

**Matlab files**: Loading and saving::

    >>> from scipy import io as spio
    >>> a = np.ones((3, 3))
    >>> spio.savemat('file.mat', {'a': a}) # savemat expects a dictionary
    >>> data = spio.loadmat('file.mat')
    >>> data['a']
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])

.. warning:: **Python / Matlab mismatches**, *eg* matlab does not represent 1D arrays
   
   ::

      >>> a = np.ones(3)
      >>> a
      array([ 1.,  1.,  1.])
      >>> spio.savemat('file.mat', {'a': a})
      >>> spio.loadmat('file.mat')['a']
      array([[ 1.,  1.,  1.]])

   Notice the difference?

|

.. Comments to make doctests pass which require an image
    >>> from matplotlib import pyplot as plt
    >>> plt.imsave('fname.png', np.array([[0]]))

**Image files**: Reading images::

    >>> from scipy import misc
    >>> misc.imread('fname.png')    # doctest: +ELLIPSIS
    array(...)
    >>> # Matplotlib also has a similar function
    >>> import matplotlib.pyplot as plt
    >>> plt.imread('fname.png')    # doctest: +ELLIPSIS
    array(...)

.. seealso::

    * Load text files: :func:`numpy.loadtxt`/:func:`numpy.savetxt`

    * Clever loading of text/csv files:
      :func:`numpy.genfromtxt`/:func:`numpy.recfromcsv`

    * Fast and efficient, but numpy-specific, binary format:
      :func:`numpy.save`/:func:`numpy.load`

    * More advanced input/output of images in scikit-image: :mod:`skimage.io`

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

.. tip::

    The :mod:`scipy.linalg` module provides standard linear algebra
    operations, relying on an underlying efficient implementation (BLAS,
    LAPACK).

* The :func:`scipy.linalg.det` function computes the determinant of a
  square matrix::

    >>> from scipy import linalg
    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> linalg.det(arr)
    -2.0
    >>> arr = np.array([[3, 2],
    ...                 [6, 4]])
    >>> linalg.det(arr) # doctest: +SKIP
    0.0
    >>> linalg.det(np.ones((3, 4)))
    Traceback (most recent call last):
    ...
    ValueError: expected square matrix

* The :func:`scipy.linalg.inv` function computes the inverse of a square
  matrix::

    >>> arr = np.array([[1, 2],
    ...                 [3, 4]])
    >>> iarr = linalg.inv(arr)
    >>> iarr
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.allclose(np.dot(arr, iarr), np.eye(2))
    True

  Finally computing the inverse of a singular matrix (its determinant is zero)
  will raise ``LinAlgError``::

    >>> arr = np.array([[3, 2],
    ...                 [6, 4]])
    >>> linalg.inv(arr)  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    ...LinAlgError: singular matrix

* More advanced operations are available, for example singular-value
  decomposition (SVD)::

    >>> arr = np.arange(9).reshape((3, 3)) + np.diag([1, 0, 1])
    >>> uarr, spec, vharr = linalg.svd(arr)

  The resulting array spectrum is::

    >>> spec    # doctest: +ELLIPSIS
    array([ 14.88982544,   0.45294236,   0.29654967])

  The original matrix can be re-composed by matrix multiplication of the outputs of
  ``svd`` with ``np.dot``::

    >>> sarr = np.diag(spec)
    >>> svd_mat = uarr.dot(sarr).dot(vharr)
    >>> np.allclose(svd_mat, arr)
    True

  SVD is commonly used in statistics and signal processing.  Many other
  standard decompositions (QR, LU, Cholesky, Schur), as well as solvers
  for linear systems, are available in :mod:`scipy.linalg`.


.. _intro_scipy_interpolate:

Interpolation: :mod:`scipy.interpolate`
---------------------------------------

:mod:`scipy.interpolate` is useful for fitting a function from experimental
data and thus evaluating points where no measure exists. The module is based
on the `FITPACK Fortran subroutines`_.

.. _`FITPACK Fortran subroutines` : http://www.netlib.org/dierckx/index.html
.. _netlib : http://www.netlib.org

By imagining experimental data close to a sine function::

    >>> measured_time = np.linspace(0, 1, 10)
    >>> noise = (np.random.random(10)*2 - 1) * 1e-1
    >>> measures = np.sin(2 * np.pi * measured_time) + noise

:class:`scipy.interpolate.interp1d` can build a linear interpolation
function::

    >>> from scipy.interpolate import interp1d
    >>> linear_interp = interp1d(measured_time, measures)

.. image:: scipy/auto_examples/images/sphx_glr_plot_interpolation_001.png
    :target: scipy/auto_examples/plot_interpolation.html
    :scale: 60
    :align: right

Then the result can be evaluated at the time of interest::

    >>> interpolation_time = np.linspace(0, 1, 50)
    >>> linear_results = linear_interp(interpolation_time)

A cubic interpolation can also be selected by providing the ``kind`` optional
keyword argument::

    >>> cubic_interp = interp1d(measured_time, measures, kind='cubic')
    >>> cubic_results = cubic_interp(interpolation_time)


:class:`scipy.interpolate.interp2d` is similar to
:class:`scipy.interpolate.interp1d`, but for 2-D arrays. Note that for
the `interp` family, the interpolation points must stay within the range
of given data points. See the summary exercise on
:ref:`summary_exercise_stat_interp` for a more advanced spline
interpolation example.



Optimization and fit: :mod:`scipy.optimize`
-------------------------------------------

Optimization is the problem of finding a numerical solution to a
minimization or equality.

.. tip::

    The :mod:`scipy.optimize` module provides algorithms for function
    minimization (scalar or multi-dimensional), curve fitting and root
    finding. ::

        >>> from scipy import optimize

Curve fitting
..............

.. Comment to make doctest pass
    >>> np.random.seed(0)

.. image:: scipy/auto_examples/images/sphx_glr_plot_curve_fit_001.png
   :target: scipy/auto_examples/plot_curve_fit.html
   :align: right
   :scale: 50

Suppose we have data on a sine wave, with some noise: ::

    >>> x_data = np.linspace(-5, 5, num=50)
    >>> y_data = 2.9 * np.sin(1.5 * x_data) + np.random.normal(size=50)


If we know that the data lies on a sine wave, but not the amplitudes
or the period, we can find those by least squares curve fitting. First we
have to define the test function to fit, here a sine with unknown
amplitude and period::

    >>> def test_func(x, a, b):
    ...     return a * np.sin(b * x)

.. image:: scipy/auto_examples/images/sphx_glr_plot_curve_fit_002.png
   :target: scipy/auto_examples/plot_curve_fit.html
   :align: right
   :scale: 50

We then use :func:`scipy.optimize.curve_fit` to find :math:`a` and :math:`b`::

    >>> params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
    >>> print(params)
    [ 3.05931973  1.45754553]

.. raw:: html

   <div style="clear: both"></div>


.. topic:: Exercise: Curve fitting of temperature data
   :class: green

    The temperature extremes in Alaska for each month, starting in January, are
    given by (in degrees Celcius)::

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


Finding the minimum of a scalar function
........................................

.. Comment to make doctest pass
    >>> np.random.seed(0)

.. image:: scipy/auto_examples/images/sphx_glr_plot_optimize_example1_001.png
   :target: scipy/auto_examples/plot_optimize_example1.html
   :align: right
   :scale: 50

Let's define the following function: ::

    >>> def f(x):
    ...     return x**2 + 10*np.sin(x)

and plot it:

.. doctest::

    >>> x = np.arange(-10, 10, 0.1)
    >>> plt.plot(x, f(x)) # doctest:+SKIP
    >>> plt.show() # doctest:+SKIP

This function has a global minimum around -1.3 and a local minimum around
3.8.

Searching for minimum can be done with
:func:`scipy.optimize.minimize`, given a starting point x0, it returns
the location of the minimum that it has found:

.. sidebar:: result type

    The result of :func:`scipy.optimize.minimize` is a compound object
    comprising all information on the convergence
::

    >>> result = optimize.minimize(f, x0=0) 
    >>> result # doctest: +ELLIPSIS
          fun: -7.9458233756...
     hess_inv: array([[ 0.0858...]])
          jac: array([ -1.19209...e-06])
      message: 'Optimization terminated successfully.'
         nfev: 18
          nit: 5
         njev: 6
       status: 0
      success: True
            x: array([-1.30644...])
    >>> result.x # The coordinate of the minimum  # doctest: +ELLIPSIS
    array([-1.30644...])

|

**Methods**:
As the function is a smooth function, gradient-descent based methods are
good options. The `lBFGS algorithm
<https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__ is a good choice
in general::


    >>> optimize.minimize(f, x0=0, method="L-BFGS-B")  # doctest: +ELLIPSIS
          fun: array([-7.94582338])
     hess_inv: <1x1 LbfgsInvHessProduct with dtype=float64>
          jac: array([ -1.42108547e-06])
      message: ...'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
         nfev: 12
          nit: 5
       status: 0
      success: True
            x: array([-1.30644013])

Note how it cost only 12 functions evaluation above to find a good value
for the minimum.

|

**Global minimum**:
A possible issue with this approach is that, if the function has local minima,
the algorithm may find these local minima instead of the
global minimum depending on the initial point x0::

    >>> res = optimize.minimize(f, x0=3, method="L-BFGS-B")
    >>> res.x
    array([ 3.83746709])

.. Comment to make doctest pass
   >>> np.random.seed(42)

If we don't know the neighborhood of the global minimum to choose the
initial point, we need to resort to costlier global optimization.  To
find the global minimum, we use :func:`scipy.optimize.basinhopping`
(added in version 0.12.0 of Scipy). It combines a local optimizer with
sampling of starting points::

   >>> optimize.basinhopping(f, 0)  # doctest: +SKIP
                     nfev: 1725
    minimization_failures: 0
                      fun: -7.9458233756152845
                        x: array([-1.30644001])
                  message: ['requested number of basinhopping iterations completed successfully']
                     njev: 575
                      nit: 100

.. seealso:

    Another available (but much less efficient) global optimizer is
    :func:`scipy.optimize.brute` (brute force optimization on a grid).

    More algorithms for different classes of global optimization problems
    exist, but this is out of the scope of :mod:`scipy`.  Some useful
    packages for global optimization are OpenOpt, IPOPT_, PyGMO_ and
    PyEvolve_.

.. note::

   :mod:`scipy` used to contain the routine `anneal`, it has been removed in
   SciPy 0.16.0.

.. _IPOPT: https://github.com/xuy/pyipopt
.. _PyGMO: http://esa.github.io/pygmo/
.. _PyEvolve: http://pyevolve.sourceforge.net/

|

**Constraints**:
We can constrain the variable to the interval
``(0, 10)`` using the "bounds" argument:

.. sidebar:: A list of bounds

   As :func:`~scipy.optimize.minimize` works in general with x
   multidimensionsal, the "bounds" argument is a list of bound on each
   dimension.

::

    >>> res = optimize.minimize(f, x0=1,
    ...                         bounds=((0, 10), ))
    >>> res.x    # doctest: +ELLIPSIS
    array([ 0.])

.. tip::

   What has happened? Why are we finding 0, which is not a mimimum of our
   function.


.. topic:: **Minimizing functions of several variables**

   To minimize over several variables, the trick is to turn them into a
   function of a multi-dimensional variable (a vector). See for instance
   the exercise on 2D minimization below.

.. note::

   :func:`scipy.optimize.minimize_scalar` is a function with dedicated
   methods to minimize functions of only one variable.

.. seealso::

   Finding minima of function is discussed in more details in the
   advanced chapter: :ref:`mathematical_optimization`.

.. topic:: Exercise: 2-D minimization
   :class: green

    .. image:: scipy/auto_examples/images/sphx_glr_plot_2d_minimization_002.png
        :target: scipy/auto_examples/plot_2d_minimization.html
        :align: right
        :scale: 50

    The six-hump camelback function

    .. math:: f(x, y) = (4 - 2.1x^2 + \frac{x^4}{3})x^2 + xy + (4y^2 - 4)y^2

    has multiple global and local minima. Find the global minima of this
    function.

    Hints:

        - Variables can be restricted to :math:`-2 < x < 2` and :math:`-1 < y < 1`.
        - Use :func:`numpy.meshgrid` and :func:`pylab.imshow` to find visually the
          regions.
        - Use :func:`scipy.optimize.minimize`, optionally trying out
          several of its `methods'.

    How many global minima are there, and what is the function value at those
    points?  What happens for an initial guess of :math:`(x, y) = (0, 0)` ?

    :ref:`solution <sphx_glr_intro_scipy_auto_examples_plot_2d_minimization.py>`


Finding the roots of a scalar function
........................................

To find a root, i.e. a point where :math:`f(x) = 0`, of the function :math:`f` above
we can use :func:`scipy.optimize.root`:

::

    >>> root = optimize.root(f, x0=1)  # our initial guess is 1
    >>> root    # The full result
        fjac: array([[-1.]])
         fun: array([ 0.])
     message: 'The solution converged.'
        nfev: 10
         qtf: array([  1.33310463e-32])
           r: array([-10.])
      status: 1
     success: True
           x: array([ 0.])
    >>> root.x  # Only the root found
    array([ 0.])

Note that only one root is found.  Inspecting the plot of :math:`f` reveals that
there is a second root around -2.5. We find the exact value of it by adjusting
our initial guess: ::

    >>> root2 = optimize.root(f, x0=-2.5)
    >>> root2.x
    array([-2.47948183])

.. note::
   
   :func:`scipy.optimize.root` also comes with a variety of algorithms,
   set via the "method" argument.

.. image:: scipy/auto_examples/images/sphx_glr_plot_optimize_example2_001.png
   :target: scipy/auto_examples/plot_optimize_example2.html
   :align: right
   :scale: 70

|


Now that we have found the minima and roots of ``f`` and used curve fitting on it,
we put all those results together in a single plot:

.. raw:: html

   <div style="clear: both"></div>


.. seealso::
   
    You can find all algorithms and functions with similar functionalities
    in the documentation of :mod:`scipy.optimize`.

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
    >>> histogram = np.histogram(samples, bins=bins, normed=True)[0]
    >>> bins = 0.5*(bins[1:] + bins[:-1])
    >>> bins
    array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5])
    >>> from scipy import stats
    >>> pdf = stats.norm.pdf(bins)  # norm is a distribution object
    
    >>> plt.plot(bins, histogram) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(bins, pdf) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]

.. image:: scipy/auto_examples/images/sphx_glr_plot_normal_distribution_001.png
    :target: scipy/auto_examples/plot_normal_distribution.html
    :scale: 70

.. sidebar:: **The distribution objects**

   :class:`scipy.stats.norm` is a distribution object: each distribution
   in :mod:`scipy.stats` is represented as an object. Here it's the
   normal distribution, and it comes with a PDF, a CDF, and much more.

If we know that the random process belongs to a given family of random
processes, such as normal processes, we can do a maximum-likelihood fit
of the observations to estimate the parameters of the underlying
distribution. Here we fit a normal process to the observed data::

    >>> loc, std = stats.norm.fit(samples)
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

    >>> stats.scoreatpercentile(samples, 50)     # doctest: +ELLIPSIS
    -0.0580280347...

Similarly, we can calculate the percentile 90::

    >>> stats.scoreatpercentile(samples, 90)     # doctest: +ELLIPSIS
    1.2315935511...

.. tip::

    The percentile is an estimator of the CDF: cumulative distribution
    function.

Statistical tests
.................

.. image:: scipy/auto_examples/images/sphx_glr_plot_t_test_001.png
    :target: scipy/auto_examples/plot_t_test.html
    :scale: 60
    :align: right

A statistical test is a decision indicator. For instance, if we have two
sets of observations, that we assume are generated from Gaussian
processes, we can use a
`T-test <https://en.wikipedia.org/wiki/Student%27s_t-test>`__ to decide
whether the means of two sets of observations are significantly different::

    >>> a = np.random.normal(0, 1, size=100)
    >>> b = np.random.normal(1, 1, size=10)
    >>> stats.ttest_ind(a, b)   # doctest: +SKIP
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

Function integrals
...................

The most generic integration routine is :func:`scipy.integrate.quad`. To
compute :math:`\int_0^{\pi / 2} sin(t) dt`::

    >>> from scipy.integrate import quad
    >>> res, err = quad(np.sin, 0, np.pi/2)
    >>> np.allclose(res, 1)   # res is the result, is should be close to 1
    True
    >>> np.allclose(err, 1 - res)  # err is an estimate of the err
    True

Other integration schemes are available:
:func:`scipy.integrate.fixed_quad`, :func:`scipy.integrate.quadrature`,
:func:`scipy.integrate.romberg`...

Integrating differential equations
...................................

:mod:`scipy.integrate` also features routines for integrating `Ordinary
Differential Equations (ODE)
<https://en.wikipedia.org/wiki/Ordinary_differential_equation>`__. In
particular, :func:`scipy.integrate.odeint` solves ODE of the form::

    dy/dt = rhs(y1, y2, .., t0,...)

As an introduction, let us solve the ODE :math:`\frac{dy}{dt} = -2 y` between 
:math:`t = 0 \dots 4`, with the  initial condition :math:`y(t=0) = 1`.
First the function computing the derivative of the position needs to be defined::

    >>> def calc_derivative(ypos, time):
    ...     return -2 * ypos

.. image:: scipy/auto_examples/images/sphx_glr_plot_odeint_simple_001.png
    :target: scipy/auto_examples/plot_odeint_simple.html
    :scale: 70
    :align: right


Then, to compute ``y`` as a function of time::

    >>> from scipy.integrate import odeint
    >>> time_vec = np.linspace(0, 4, 40)
    >>> y = odeint(calc_derivative, y0=1, t=time_vec)

.. raw:: html

   <div style="clear: both"></div>

Let us integrate a more complex ODE: a `damped
spring-mass oscillator
<https://en.wikipedia.org/wiki/Harmonic_oscillator#Damped_harmonic_oscillator>`__.
The position of a mass attached to a spring obeys the 2nd order *ODE*
:math:`y'' + 2 \varepsilon \omega_0  y' + \omega_0^2 y = 0` with 
:math:`\omega_0^2 = k/m` with :math:`k` the spring constant, :math:`m` the mass
and :math:`\varepsilon = c/(2 m \omega_0)` with :math:`c` the damping coefficient. We set::

    >>> mass = 0.5  # kg
    >>> kspring = 4  # N/m
    >>> cviscous = 0.4  # N s/m

Hence::

    >>> eps = cviscous / (2 * mass * np.sqrt(kspring/mass))
    >>> omega = np.sqrt(kspring / mass)

The system is underdamped, as::

    >>> eps < 1
    True

For :func:`~scipy.integrate.odeint`, the 2nd order equation
needs to be transformed in a system of two first-order equations for the
vector :math:`Y = (y, y')`: the function computes the
velocity and acceleration::
    
    >>> def calc_deri(yvec, time, eps, omega):
    ...     return (yvec[1], -eps * omega * yvec[1] - omega **2 * yvec[0])

.. image:: scipy/auto_examples/images/sphx_glr_plot_odeint_damped_spring_mass_001.png
    :target: scipy/auto_examples/plot_odeint_damped_spring_mass.html
    :scale: 70
    :align: right

Integration of the system follows::

    >>> time_vec = np.linspace(0, 10, 100)
    >>> yinit = (1, 0)
    >>> yarr = odeint(calc_deri, yinit, time_vec, args=(eps, omega))

.. raw:: html

   <div style="clear: both"></div>


.. tip::

    :func:`scipy.integrate.odeint` uses the LSODA (Livermore Solver for
    Ordinary Differential equations with Automatic method switching for stiff
    and non-stiff problems), see the `ODEPACK Fortran library`_ for more
    details.

.. _`ODEPACK Fortran library` : http://people.sc.fsu.edu/~jburkardt/f77_src/odepack/odepack.html

.. seealso:: **Partial Differental Equations**

    There is no Partial Differential Equations (PDE) solver in Scipy.
    Some Python packages for solving PDE's are available, such as fipy_
    or SfePy_.

.. _fipy: http://www.ctcms.nist.gov/fipy/
.. _SfePy: http://sfepy.org/doc/

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

    >>> from scipy import fftpack
    >>> sig_fft = fftpack.fft(sig) # doctest:+SKIP
    >>> freqs = fftpack.fftfreq(sig.size, d=time_step) # doctest:+SKIP


.. |signal_fig| image:: scipy/auto_examples/images/sphx_glr_plot_fftpack_001.png
    :target: scipy/auto_examples/plot_fftpack.html
    :scale: 60

.. |fft_fig| image:: scipy/auto_examples/images/sphx_glr_plot_fftpack_002.png
    :target: scipy/auto_examples/plot_fftpack.html
    :scale: 60

===================== =====================
|signal_fig|          |fft_fig|
===================== =====================
**Signal**            **FFT**
===================== =====================

As the signal comes from a real function, the Fourier transform is
symmetric.

The peak signal frequency can be found with ``freqs[power.argmax()]``

.. image:: scipy/auto_examples/images/sphx_glr_plot_fftpack_003.png
    :target: scipy/auto_examples/plot_fftpack.html
    :scale: 60
    :align: right


Setting the Fourrier component above this frequency to zero and inverting
the FFT with :func:`scipy.fftpack.ifft`, gives a filtered signal.

.. note::

   The code of this example can be found :ref:`here <sphx_glr_intro_scipy_auto_examples_plot_fftpack.py>`

.. topic:: `numpy.fft`

   Numpy also has an implementation of FFT (:mod:`numpy.fft`). However,
   the scipy one
   should be preferred, as it uses more efficient underlying implementations.

|

**Fully worked examples:**

.. |periodicity_finding| image:: scipy/auto_examples/solutions/images/sphx_glr_plot_periodicity_finder_001.png
    :scale: 50
    :target: scipy/auto_examples/solutions/plot_periodicity_finder.html

.. |image_blur| image:: scipy/auto_examples/solutions/images/sphx_glr_plot_image_blur_002.png
    :scale: 50
    :target: scipy/auto_examples/solutions/plot_image_blur.html

=================================================================================================================== ===================================================================================================================
Crude periodicity finding (:ref:`link <sphx_glr_intro_scipy_auto_examples_solutions_plot_periodicity_finder.py>`)   Gaussian image blur (:ref:`link <sphx_glr_intro_scipy_auto_examples_solutions_plot_image_blur.py>`)
=================================================================================================================== ===================================================================================================================
|periodicity_finding|                                                                                               |image_blur|
=================================================================================================================== ===================================================================================================================

|

.. topic:: Exercise: Denoise moon landing image
   :class: green

   .. image:: ../data/moonlanding.png
     :scale: 70

   1. Examine the provided image :download:`moonlanding.png
      <../data/moonlanding.png>`, which is heavily contaminated with periodic
      noise. In this exercise, we aim to clean up the noise using the
      Fast Fourier Transform.

   2. Load the image using :func:`pylab.imread`.

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

.. image:: scipy/auto_examples/images/sphx_glr_plot_resample_001.png
    :target: scipy/auto_examples/plot_resample.html
    :scale: 65
    :align: right


**Resampling** :func:`scipy.signal.resample`: resample a signal to `n`
points using FFT. ::

  >>> t = np.linspace(0, 5, 100)
  >>> x = np.sin(t)

  >>> from scipy import signal
  >>> x_resampled = signal.resample(x, 25)

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


.. image:: scipy/auto_examples/images/sphx_glr_plot_detrend_001.png
    :target: scipy/auto_examples/plot_detrend.html
    :scale: 65
    :align: right

**Detrending** :func:`scipy.signal.detrend`: remove linear trend from signal::

  >>> t = np.linspace(0, 5, 100)
  >>> x = t + np.random.normal(size=100)

  >>> from scipy import signal
  >>> x_detrended = signal.detrend(x)

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

.. |chirp_fig| image:: scipy/auto_examples/images/sphx_glr_plot_spectrogram_001.png
    :target: scipy/auto_examples/plot_spectrogram.html
    :scale: 45

.. |spectrogram_fig| image:: scipy/auto_examples/images/sphx_glr_plot_spectrogram_002.png
    :target: scipy/auto_examples/plot_spectrogram.html
    :scale: 45

.. |psd_fig| image:: scipy/auto_examples/images/sphx_glr_plot_spectrogram_003.png
    :target: scipy/auto_examples/plot_spectrogram.html
    :scale: 45

|chirp_fig| |spectrogram_fig| |psd_fig|

Image manipulation: :mod:`scipy.ndimage`
-----------------------------------------

.. include:: image_processing/image_processing.rst
    :start-line: 1


Summary exercises on scientific computing
-----------------------------------------

The summary exercises use mainly Numpy, Scipy and Matplotlib. They provide some
real-life examples of scientific computing with Python. Now that the basics of
working with Numpy and Scipy have been introduced, the interested user is
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

.. include:: scipy/auto_examples/index.rst
    :start-line: 1


.. seealso:: **References to go further**

   * Some chapters of the `advanced <advanced_topics_part>`__ and the
     `packages and applications <applications_part>`__ parts of the scipy
     lectures

   * The `scipy cookbook <http://scipy-cookbook.readthedocs.io>`__


