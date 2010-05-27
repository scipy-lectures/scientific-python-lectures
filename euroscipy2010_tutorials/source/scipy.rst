Scipy : high-level scientific computing
=========================================

.. topic:: Scipy

    The ``scipy`` package contains various toolboxes dedicated to common
    issues in scientific computing.

Interpolation: ``scipy.interpolate``
---------------------------------------
The interpolate module is very useful for fitting a function from
experimental data in order to evaluate points where no measure exists.
An illustration will first be given with maximum wind speeds measured over
21 years at the Sprogø meteorological station located in Denmark. The
interested readers are then invited to compute results from raw data at
the end of the section.

The annual wind speeds maxima are loaded by using numpy::

    >>> import numpy as N
    >>> max_speeds = N.load('data/max-speeds.npy')
    >>> years_nb = max_speeds.shape[0]

In order to guess extreme wind speed, the chosen model is based on the
cumulative probability function of annual maxima. Thus, the cumulative
probabilty ``p_i`` for a given year ``i`` is defined as
``p_i = i/(N+1)`` with ``N = 21``, the number of measured years. As
a result, the cumulative probability of the measured data will be::

    >>> cprob = (N.arange(years_nb, dtype=N.float32) + 1)/(years_nb + 1)

and the corresponding wind speeds are::

    >>> sorted_max_speeds = N.sort(max_speeds)

From there, a function giving a wind speed from a probability value
is wanted. The interpolate module of scipy provides the ``UnivariateSpline``
class which can represent a spline from points. The default behavior
is to build a spline of degree 3 and points can have different weights
according to their reliability. Variantes are ``InterpolatedUnivariateSpline``
and ``LSQUnivariateSpline`` on which errors checking is going to change.
In case a 2D spline is wanted, the ``BivariateSpline`` class family
is provided. All thoses classes for 1D and 2D splines use the FITPACK
Fortran subroutines, that's why a lower library access is available
through the ``splrep`` and ``splev`` functions for respectively
representing and evaluating a spline. Moreover interpolation
functions without the use of FITPACK parameters are provided for simpler
use (see ``interp1d``, ``interp2d``, ``barycentric_interpolate`` and
so on).

For the Sprogø maxima wind speeds, the ``UnivariateSpline`` will be
used because a spline of degree 3 seems to correctly fit the data::

    >>> from scipy.interpolate import UnivariateSpline
    >>> speed_spline = UnivariateSpline(cprob, sorted_max_speeds)

The function is now going to be evaluated from the full range
of probabilties::

    >>> nprob = N.linspace(0, 1, 1e2)
    >>> fitted_max_speeds = speed_spline(nprob)

In the current model, the maximum wind speed occuring every 50 years
V_50 is defined as the upper 2% quantile. As a result, the cumulative
probability value will be::

    >>> fifty_prob = 1. - 0.02

So the storm wind speed occuring every 50 years can be guessed by::

    >>> fifty_wind = speed_spline(fifty_prob)
    >>> fifty_wind
    array([ 32.97989825])

The results are now gathered on a Matplotlib figure.

.. image:: cumulative-wind-speed-prediction.png
   :align: center

The interested readers are now invited to make an exercice by
using the measured wind speeds over 21 years available in the file
``sprog-windspeeds.npy``.

* The first step will be to find the annual maxima by using numpy
  and plot them as a matplotlib bar figure.

.. image:: sprog-annual-maxima.png
   :align: center

* The second step will be to use the Gumbell distribution on cumulative
  probabilities ``p_i`` defined as ``-log( -log(p_i) )`` for fitting
  a linear function (remember that you can define the degree of the
  ``UnivariateSpline``). Plotting the annual maxima versus the Gumbell
  distribution should give you the following figure.

.. image:: gumbell-wind-speed-prediction.png
   :align: center

* The last step will be to find 34.23 m/s for the maximum wind speed
  occuring every 50 years.


