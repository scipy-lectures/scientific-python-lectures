Scipy : high-level scientific computing
=========================================

:authors: Adrien Chauve, Andre Espaze, Emmanuelle Gouillart,
          Gaël Varoquaux, Ralf Gommers

..
    >>> import numpy as np
    >>> np.random.seed(0)

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

=========================== ===============================================
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
=========================== ===============================================

They all depend on :mod:`numpy`, but are mostly independent of each other. The
standard way of importing Numpy and these Scipy modules is::

    >>> import numpy as np
    >>> from scipy import stats  # same for other sub-modules

The main ``scipy`` namespace mostly contains functions that are really
numpy functions (try ``scipy.cos is np.cos``). Those are exposed for historical
reasons only; there's usually no reason to use ``import scipy`` in your code.


File input/output: :mod:`scipy.io`
----------------------------------

* Loading and saving matlab files::

    >>> from scipy import io as spio
    >>> a = np.ones((3, 3))
    >>> spio.savemat('file.mat', {'a': a}) # savemat expects a dictionary
    >>> data = spio.loadmat('file.mat', struct_as_record=True)
    >>> data['a']
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])

* Reading images::

    >>> from scipy import misc
    >>> misc.imread('fname.png')
    >>> # Matplotlib also has a similar function
    >>> import matplotlib.pyplot as plt
    >>> plt.imread('fname.png')

See also:

    * Load text files: :func:`numpy.loadtxt`/:func:`numpy.savetxt`

    * Clever loading of text/csv files:
      :func:`numpy.genfromtxt`/:func:`numpy.recfromcsv`

    * Fast and efficient, but numpy-specific, binary format:
      :func:`numpy.save`/:func:`numpy.load`



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
-----------------------------------------------

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
    >>> linalg.det(arr)
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
    >>> linalg.inv(arr)
    Traceback (most recent call last):
    ...
    LinAlgError: singular matrix

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


Fast Fourier transforms: :mod:`scipy.fftpack`
----------------------------------------------

The :mod:`scipy.fftpack` module allows to compute fast Fourier transforms.
As an illustration, a (noisy) input signal may look like::

    >>> time_step = 0.02
    >>> period = 5.
    >>> time_vec = np.arange(0, 20, time_step)
    >>> sig = np.sin(2 * np.pi / period * time_vec) + \
    ...       0.5 * np.random.randn(time_vec.size)

The observer doesn't know the signal frequency, only
the sampling time step of the signal ``sig``. The signal
is supposed to come from a real function so the Fourier transform
will be symmetric.
The :func:`scipy.fftpack.fftfreq` function will generate the sampling frequencies and
:func:`scipy.fftpack.fft` will compute the fast Fourier transform::

    >>> from scipy import fftpack
    >>> sample_freq = fftpack.fftfreq(sig.size, d=time_step)
    >>> sig_fft = fftpack.fft(sig)

Because the resulting power is symmetric, only the positive part of the
spectrum needs to be used for finding the frequency::

    >>> pidxs = np.where(sample_freq > 0)
    >>> freqs = sample_freq[pidxs]
    >>> power = np.abs(sig_fft)[pidxs]

.. plot:: pyplots/fftpack_frequency.py
    :scale: 70

The signal frequency can be found by::

    >>> freq = freqs[power.argmax()]
    >>> np.allclose(freq, 1./period)  # check that correct freq is found
    True

Now the high-frequency noise will be removed from the Fourier transformed
signal::

    >>> sig_fft[np.abs(sample_freq) > freq] = 0

The resulting filtered signal can be computed by the
:func:`scipy.fftpack.ifft` function::

    >>> main_sig = fftpack.ifft(sig_fft)

The result can be viewed with::

    >>> import pylab as plt
    >>> plt.figure()
    >>> plt.plot(time_vec, sig)
    >>> plt.plot(time_vec, main_sig, linewidth=3)
    >>> plt.xlabel('Time [s]')
    >>> plt.ylabel('Amplitude')

.. plot:: pyplots/fftpack_signals.py
    :scale: 70

.. topic:: `numpy.fft`

   Numpy also has an implementation of FFT (:mod:`numpy.fft`). However,
   in general the scipy one
   should be preferred, as it uses more efficient underlying implementations.

.. topic:: Worked example: Crude periodicity finding

    .. plot:: intro/solutions/periodicity_finder.py

.. topic:: Worked example: Gaussian image blur

    Convolution:

    .. math::

        f_1(t) = \int dt'\, K(t-t') f_0(t')

    .. math::

        \tilde{f}_1(\omega) = \tilde{K}(\omega) \tilde{f}_0(\omega)

    .. plot:: intro/solutions/image_blur.py

.. topic:: Exercise: Denoise moon landing image
   :class: green

   .. image:: ../data/moonlanding.png
     :scale: 70

   1. Examine the provided image moonlanding.png, which is heavily
      contaminated with periodic noise. In this exercise, we aim to clean up
      the noise using the Fast Fourier Transform.

   2. Load the image using :func:`pylab.imread`.

   3. Find and use the 2-D FFT function in :mod:`scipy.fftpack`, and plot the
      spectrum (Fourier transform of) the image. Do you have any trouble
      visualising the spectrum? If so, why?

   4. The spectrum consists of high and low frequency components. The noise is
      contained in the high-frequency part of the spectrum, so set some of
      those components to zero (use array slicing).

   5. Apply the inverse Fourier transform to see the resulting image.


Optimization and fit: :mod:`scipy.optimize`
-------------------------------------------

Optimization is the problem of finding a numerical solution to a
minimization or equality.

The :mod:`scipy.optimize` module provides useful algorithms for function
minimization (scalar or multi-dimensional), curve fitting and root
finding. ::

    >>> from scipy import optimize


**Finding the minimum of a scalar function**

Let's define the following function: ::

    >>> def f(x):
    ...     return x**2 + 10*np.sin(x)

and plot it:

.. doctest::

    >>> x = np.arange(-10, 10, 0.1)
    >>> plt.plot(x, f(x)) # doctest:+SKIP
    >>> plt.show() # doctest:+SKIP

.. plot:: pyplots/scipy_optimize_example1.py

This function has a global minimum around -1.3 and a local minimum around 3.8.

The general and efficient way to find a minimum for this function is to
conduct a gradient descent starting from a given initial point. The BFGS
algorithm is a good way of doing this::

    >>> optimize.fmin_bfgs(f, 0)
    Optimization terminated successfully.
	     Current function value: -7.945823
	     Iterations: 5
	     Function evaluations: 24
	     Gradient evaluations: 8
    array([-1.30644003])

A possible issue with this approach is that, if the function has local minima
the algorithm may find these local minima instead of the
global minimum depending on the initial point: ::

    >>> optimize.fmin_bfgs(f, 3, disp=0)
    array([ 3.83746663])

If we don't know the neighborhood of the global minimum to choose the initial
point, we need to resort to costlier global optimization.  To find the global
minimum, the simplest algorithm is the brute force algorithm, in which the
function is evaluated on each point of a given grid: ::

    >>> grid = (-10, 10, 0.1)
    >>> xmin_global = optimize.brute(f, (grid,))
    >>> xmin_global
    array([-1.30641113])

For larger grid sizes, :func:`scipy.optimize.brute` becomes quite slow.
:func:`scipy.optimize.anneal`
provides an alternative, using simulated annealing. More efficient algorithms
for different classes of global optimization problems exist, but this is out of
the scope of ``scipy``.  Some useful packages for global optimization are
OpenOpt_, IPOPT_, PyGMO_ and PyEvolve_.

.. _OpenOpt: http://openopt.org/Welcome
.. _IPOPT: https://github.com/xuy/pyipopt
.. _PyGMO: http://pagmo.sourceforge.net/pygmo/index.html
.. _PyEvolve: http://pyevolve.sourceforge.net/

To find the local minimum, let's constraint the variable to the interval
``(0, 10)`` using :func:`scipy.optimize.fminbound`: ::

    >>> xmin_local = optimize.fminbound(f, 0, 10)    # doctest: +ELLIPSIS
    >>> xmin_local
    3.8374671...

.. note::

   Finding minima of function is discussed in more details in the
   advanced chapter: :ref:`mathematical_optimization`.

**Finding the roots of a scalar function**

To find a root, i.e. a point where ``f(x) = 0``, of the function ``f`` above
we can use for example :func:`scipy.optimize.fsolve`: ::

    >>> root = optimize.fsolve(f, 1)  # our initial guess is 1
    >>> root
    array([ 0.])

Note that only one root is found.  Inspecting the plot of ``f`` reveals that
there is a second root around -2.5. We find the exact value of it by adjusting
our initial guess: ::

    >>> root2 = optimize.fsolve(f, -2.5)
    >>> root2
    array([-2.47948183])

**Curve fitting**

Suppose we have data sampled from ``f`` with some noise: ::

    >>> xdata = np.linspace(-10, 10, num=20)
    >>> ydata = f(xdata) + np.random.randn(xdata.size)

Now if we know the functional form of the function from which the samples were
drawn (``x^2 + sin(x)`` in this case) but not the amplitudes of the terms, we
can find those by least squares curve fitting. First we have to define the
function to fit::

    >>> def f2(x, a, b):
    ...     return a*x**2 + b*np.sin(x)

Then we can use :func:`scipy.optimize.curve_fit` to find ``a`` and ``b``: ::

    >>> guess = [2, 2]
    >>> params, params_covariance = optimize.curve_fit(f2, xdata, ydata, guess)
    >>> params
    array([ 0.99925147,  9.76065551])

Now we have found the minima and roots of ``f`` and used curve fitting on it,
we put all those resuls together in a single plot:

.. plot:: pyplots/scipy_optimize_example2.py

.. note::

   In Scipy >= 0.11 unified interfaces to all minimization and root
   finding algorithms are available: :func:`scipy.optimize.minimize`,
   :func:`scipy.optimize.minimize_scalar` and
   :func:`scipy.optimize.root`.  They allow comparing various algorithms
   easily through the ``method`` keyword.

You can find algorithms with the same functionalities for multi-dimensional
problems in :mod:`scipy.optimize`.

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

.. topic:: Exercise: 2-D minimization
   :class: green

    .. plot:: pyplots/scipy_optimize_sixhump.py

    The six-hump camelback function

    .. math:: f(x, y) = (4 - 2.1x^2 + \frac{x^4}{3})x^2 + xy + (4y^2 - 4)y^2

    has multiple global and local minima. Find the global minima of this
    function.

    Hints:

        - Variables can be restricted to ``-2 < x < 2`` and ``-1 < y < 1``.
        - Use :func:`numpy.meshgrid` and :func:`pylab.imshow` to find visually the
          regions.
        - Use :func:`scipy.optimize.fmin_bfgs` or another multi-dimensional
          minimizer.

    How many global minima are there, and what is the function value at those
    points?  What happens for an initial guess of ``(x, y) = (0, 0)``?

See the summary exercise on :ref:`summary_exercise_optimize` for another, more
advanced example.


Statistics and random numbers: :mod:`scipy.stats`
-------------------------------------------------

The module :mod:`scipy.stats` contains statistical tools and probabilistic
descriptions of random processes. Random number generators for various
random process can be found in :mod:`numpy.random`.

Histogram and probability density function
...............................................

Given observations of a random process, their histogram is an estimator of
the random process's PDF (probability density function): ::

    >>> a = np.random.normal(size=1000)
    >>> bins = np.arange(-4, 5)
    >>> bins
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> histogram = np.histogram(a, bins=bins, normed=True)[0]
    >>> bins = 0.5*(bins[1:] + bins[:-1])
    >>> bins
    array([-3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5])
    >>> from scipy import stats
    >>> b = stats.norm.pdf(bins)  # norm is a distribution

.. sourcecode:: ipython

    In [1]: pl.plot(bins, histogram)
    In [2]: pl.plot(bins, b)

.. plot:: pyplots/normal_distribution.py
    :scale: 70

If we know that the random process belongs to a given family of random
processes, such as normal processes, we can do a maximum-likelihood fit
of the observations to estimate the parameters of the underlying
distribution. Here we fit a normal process to the observed data::

    >>> loc, std = stats.norm.fit(a)
    >>> loc     # doctest: +ELLIPSIS
    -0.045256707490...
    >>> std     # doctest: +ELLIPSIS
    0.9870331586690...

.. topic:: Exercise: Probability distributions
   :class: green

   Generate 1000 random variates from a gamma distribution with a shape
   parameter of 1, then plot a histogram from those samples.  Can you plot the
   pdf on top (it should match)?

   Extra: the distributions have a number of useful methods. Explore them by
   reading the docstring or by using IPython tab completion.  Can you find the
   shape parameter of 1 back by using the ``fit`` method on your random
   variates?


Percentiles
.............

The median is the value with half of the observations below, and half
above::

    >>> np.median(a)     # doctest: +ELLIPSIS
    -0.058028034...

It is also called the percentile 50, because 50% of the observation are
below it::

    >>> stats.scoreatpercentile(a, 50)     # doctest: +ELLIPSIS
    -0.0580280347...

Similarly, we can calculate the percentile 90::

    >>> stats.scoreatpercentile(a, 90)     # doctest: +ELLIPSIS
    1.231593551...

The percentile is an estimator of the CDF: cumulative distribution
function.

Statistical tests
...................

A statistical test is a decision indicator. For instance, if we have two
sets of observations, that we assume are generated from Gaussian
processes, we can use a
`T-test <http://en.wikipedia.org/wiki/Student%27s_t-test>`__ to decide
whether the two sets of observations are significantly different::

    >>> a = np.random.normal(0, 1, size=100)
    >>> b = np.random.normal(1, 1, size=10)
    >>> stats.ttest_ind(a, b)   # doctest: +ELLIPSIS
    (-3.75832707..., 0.00027786...)

The resulting output is composed of:

    * The T statistic value: it is a number the sign of which is
      proportional to the difference between the two random processes and
      the magnitude is related to the significance of this difference.

    * the *p value*: the probability of both processes being identical. If it
      is close to 1, the two process are almost certainly identical.
      The closer it is to zero, the more likely it is that the processes
      have different means.


Interpolation: :mod:`scipy.interpolate`
----------------------------------------

The :mod:`scipy.interpolate` is useful for fitting a function from experimental
data and thus evaluating points where no measure exists. The module is based
on the `FITPACK Fortran subroutines`_ from the netlib_ project.

.. _`FITPACK Fortran subroutines` : http://www.netlib.org/dierckx/index.html
.. _netlib : http://www.netlib.org

By imagining experimental data close to a sine function::

    >>> measured_time = np.linspace(0, 1, 10)
    >>> noise = (np.random.random(10)*2 - 1) * 1e-1
    >>> measures = np.sin(2 * np.pi * measured_time) + noise

The :class:`scipy.interpolate.interp1d` class can build a linear
interpolation function::

    >>> from scipy.interpolate import interp1d
    >>> linear_interp = interp1d(measured_time, measures)

Then the :obj:`scipy.interpolate.linear_interp` instance needs to be
evaluated at the time of interest::

    >>> computed_time = np.linspace(0, 1, 50)
    >>> linear_results = linear_interp(computed_time)

A cubic interpolation can also be selected by providing the ``kind`` optional
keyword argument::

    >>> cubic_interp = interp1d(measured_time, measures, kind='cubic')
    >>> cubic_results = cubic_interp(computed_time)

The results are now gathered on the following Matplotlib figure:

.. plot:: pyplots/scipy_interpolation.py

:class:`scipy.interpolate.interp2d` is similar to
:class:`scipy.interpolate.interp1d`, but for 2-D arrays. Note that for
the `interp` family, the computed time must stay within the measured
time range. See the summary exercise on
:ref:`summary_exercise_stat_interp` for a more advance spline
interpolation example.


Numerical integration: :mod:`scipy.integrate`
------------------------------------------------

The most generic integration routine is :func:`scipy.integrate.quad`::

    >>> from scipy.integrate import quad
    >>> res, err = quad(np.sin, 0, np.pi/2)
    >>> np.allclose(res, 1)
    True
    >>> np.allclose(err, 1 - res)
    True

Others integration schemes are available with ``fixed_quad``,
``quadrature``, ``romberg``.

:mod:`scipy.integrate` also features routines for integrating Ordinary
Differential Equations (ODE). In particular, :func:`scipy.integrate.odeint`
is a general-purpose integrator using LSODA (Livermore Solver for
Ordinary Differential equations with Automatic method switching
for stiff and non-stiff problems), see the `ODEPACK Fortran library`_
for more details.

.. _`ODEPACK Fortran library` : http://people.sc.fsu.edu/~jburkardt/f77_src/odepack/odepack.html

``odeint`` solves first-order ODE systems of the form::

    dy/dt = rhs(y1, y2, .., t0,...)

As an introduction, let us solve the ODE ``dy/dt = -2y`` between ``t =
0..4``, with the  initial condition ``y(t=0) = 1``. First the function
computing the derivative of the position needs to be defined::

    >>> def calc_derivative(ypos, time, counter_arr):
    ...     counter_arr += 1
    ...     return -2 * ypos
    ...

An extra argument ``counter_arr`` has been added to illustrate that the
function may be called several times for a single time step, until solver
convergence. The counter array is defined as::

    >>> counter = np.zeros((1,), dtype=np.uint16)

The trajectory will now be computed::

    >>> from scipy.integrate import odeint
    >>> time_vec = np.linspace(0, 4, 40)
    >>> yvec, info = odeint(calc_derivative, 1, time_vec,
    ...                     args=(counter,), full_output=True)

Thus the derivative function has been called more than 40 times
(which was the number of time steps)::

    >>> counter
    array([129], dtype=uint16)

and the cumulative number of iterations for each of the 10 first time steps
can be obtained by::

    >>> info['nfe'][:10]
    array([31, 35, 43, 49, 53, 57, 59, 63, 65, 69], dtype=int32)

Note that the solver requires more iterations for the first time step.
The solution ``yvec`` for the trajectory can now be plotted:

  .. plot:: pyplots/odeint_introduction.py
    :scale: 70

Another example with :func:`scipy.integrate.odeint` will be a damped
spring-mass oscillator
(2nd order oscillator). The position of a mass attached to a spring obeys
the 2nd order ODE ``y'' + 2 eps wo  y' + wo^2 y = 0`` with ``wo^2 = k/m``
with ``k`` the spring constant, ``m`` the mass and ``eps=c/(2 m wo)``
with ``c`` the damping coefficient.
For this example, we choose the parameters as::

    >>> mass = 0.5  # kg
    >>> kspring = 4  # N/m
    >>> cviscous = 0.4  # N s/m

so the system will be underdamped, because::

    >>> eps = cviscous / (2 * mass * np.sqrt(kspring/mass))
    >>> eps < 1
    True

For the :func:`scipy.integrate.odeint` solver the 2nd order equation needs to be transformed in a
system of two first-order equations for the vector ``Y=(y, y')``.  It will
be convenient to define ``nu = 2 eps * wo = c / m`` and ``om = wo^2 = k/m``::

    >>> nu_coef = cviscous / mass
    >>> om_coef = kspring / mass

Thus the function will calculate the velocity and acceleration by::

    >>> def calc_deri(yvec, time, nuc, omc):
    ...     return (yvec[1], -nuc * yvec[1] - omc * yvec[0])
    ...
    >>> time_vec = np.linspace(0, 10, 100)
    >>> yarr = odeint(calc_deri, (1, 0), time_vec, args=(nu_coef, om_coef))

The final position and velocity are shown on the following Matplotlib figure:

.. plot:: pyplots/odeint_damped_spring_mass.py
    :scale: 70

There is no Partial Differential Equations (PDE) solver in Scipy.
Some Python packages for solving PDE's are available, such as fipy_ or SfePy_.

.. _fipy: http://www.ctcms.nist.gov/fipy/
.. _SfePy: http://code.google.com/p/sfepy/


Signal processing: :mod:`scipy.signal`
---------------------------------------

::

    >>> from scipy import signal

* :func:`scipy.signal.detrend`: remove linear trend from signal::

    t = np.linspace(0, 5, 100)
    x = t + np.random.normal(size=100)

    pl.plot(t, x, linewidth=3)
    pl.plot(t, signal.detrend(x), linewidth=3)

  .. plot:: pyplots/demo_detrend.py
    :scale: 70

* :func:`scipy.signal.resample`: resample a signal to `n` points using FFT. ::

    t = np.linspace(0, 5, 100)
    x = np.sin(t)

    pl.plot(t, x, linewidth=3)
    pl.plot(t[::2], signal.resample(x, 50), 'ko')

  .. plot:: pyplots/demo_resample.py
    :scale: 70

  .. only:: latex

     Notice how on the side of the window the resampling is less accurate
     and has a rippling effect.

* :mod:`scipy.signal` has many window functions: :func:`scipy.signal.hamming`,
  :func:`scipy.signal.bartlett`, :func:`scipy.signal.blackman`...

* :mod:`scipy.signal` has filtering (median filter :func:`scipy.signal.medfilt`,
  Wiener :func:`scipy.signal.wiener`), but we will
  discuss this in the image section.


Image processing: :mod:`scipy.ndimage`
------------------------------------------

.. include:: image_processing/image_processing.rst


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
