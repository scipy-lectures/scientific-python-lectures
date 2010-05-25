Scipy : high-level scientific computing
=========================================

..
    >>> import numpy as np

.. topic:: Scipy

    .. image:: square_wheels_edited.png
       :align: right

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

    >>> import scipy

Special functions: ``scipy.special``
----------------------------------------

A library of special functions.

The most standard functions are found in numpy 

.. sourcecode:: ipython

    [In 57]: np.cos, np.sinc, np.tanh
    (<ufunc 'cos'>, <function sinc at 0x8adb6bc>, <ufunc 'tanh'>)

more specialized functions are in ``scipy.special``.

**Exemple** : first-type Bessel function

.. image:: bessel-eq.png
   :align: left

.. image:: bessel.png
   :align: center

(source : linux mag HS40
http://www.linuxmag-france.org/produit.php?produit=609, Gaël Varoquaux)

::

    >>> from scipy.special import jn
    >>> index = 1
    >>> x = np.linspace(0, 10, 200)
    >>> y = jn(index, x)

And many other functions: ``gamma, beta, erf, airy, legendre,
fonctions elliptiques``, etc.

Linear algebra: ``scipy.linalg``
------------------------------------

Standard linear algebra operations.

**Basic operations** ::

    >>> from scipy import linalg
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a = scipy.mat(a) # array to matrix
    >>> linalg.eigvals(a) # eigenvalues
    array([-0.37228132+0.j,  5.37228132+0.j])
    >>> linalg.det(a) # determinant
    -2.0
    >>> a.I # inverse
    matrix([[-2. ,  1. ],
            [ 1.5, -0.5]])
    >>> linalg.inv(a) # other possibility for inverse
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])

... and more advanced operations. Example: singular-value decomposition
(SVD) ::
 
    >>> A = scipy.mat('[1 3 2; 1 2 3]') #non-square matrix
    >>> U,s,Vh = linalg.svd(A)
    >>> s # A's spectrum
    array([ 5.19615242,  1.        ])
    >>> print U
    [[-0.70710678 -0.70710678]
     [-0.70710678  0.70710678]]


SVD is commonly used in statistics or signal processing::

    >>> l = scipy.lena()
    >>> rows, weight, columns = linalg.svd(l, full_matrices=False)
    >>> white_lena = np.dot(rows, columns)
    >>> weight[20:]=0
    >>> W = scipy.mat(linalg.diagsvd(weight,512,512))
    >>> filtered_lena = np.dot(rows, np.dot(W, columns))

.. image:: svd_lena.png
   :align: center


Many other standard decompositions (QR, LU, Cholesky, Schur), as well as
solvers for linear systems, are available in ``scipy.linalg``.


Interpolation: ``scipy.interpolate``
---------------------------------------

::

    >>> import scipy.interpolate
    >>> x = np.linspace(0, 1, 10)
    >>> y = np.sin(2*np.pi*x)
    >>> linear_interp = scipy.interpolate.interp1d(x, y) # default is linear interpolation
    >>> cubic_interp = scipy.interpolate.interp1d(x, y, kind='cubic') # cubic interpolator
    >>> fine_x = np.linspace(0, 1, 50)
    >>> y_fromlinearinterp = linear_interp(fine_x)
    >>> y_fromcubicinterp = cubic_interp(fine_x)

.. image:: interpolation.png
   :align: center

.. sourcecode:: ipython

    In [246]: plot(x, y, 'o', ms=6, label='original data')
    Out[246]: [<matplotlib.lines.Line2D object at 0x9ba55ec>]
    In [247]: plot(fine_x, y_fromlinearinterp, '-', lw=2,
    label='linear interpolation')
    Out[247]: [<matplotlib.lines.Line2D object at 0xc8300cc>]
    In [248]: plot(fine_x, y_fromcubicinterp, '-', lw=2, label='cubic
    interpolation')
    Out[248]: [<matplotlib.lines.Line2D object at 0xc616b4c>]


For spline interpolation, see ``scipy.interpolate.splrep`` and
``scipy.interpolate.splev``.

``scipy.interpolate.interp2d``  is similar to ``interp1d``, but for 2-D
arrays.

Integration: ``scipy.integrate``
-----------------------------------

Numerical integration routines. ``scipy.integrate.quad`` is the more
generic one.

.. sourcecode:: ipython

    In [92]: from scipy.integrate import quad
    In [93]: quad(np.sin, 0, np.pi/2)
    Out[93]: (0.99999999999999989, 1.1102230246251564e-14)
    In [94]: # small error
    In [95]: 1 - Out[93][0] 
    Out[95]: 1.1102230246251565e-16


**Ordinary differential equations (ODE)**

``scipy.integrate`` also features routines for ODE integration. In
particular, ``scipy.integrate.odeint`` is a general-purpose integrator,
that is suited for stiff and non-stiff problems. ``odeint`` solves
first-order ODE systems of the form


``dy1/dt = rhs1(y1, y2, .., t0,...)``

``dy2/dt = rhs2(y1, y2, .., t0,...)``

...

Example : let us solve the ODE ``dy/dt = -2y`` between ``t=0..10``, with
initial condition
``y(t=0)=1``.::

    >>> from scipy.integrate import odeint
    >>> def rhs(y, t): #ODE right hand side
    ...     return -2*y
    ...
    >>> t = np.linspace(0, 10, 100)
    >>> y = odeint(rhs, 1, t) # solution 


.. image:: odeint.png
   :align: center

.. sourcecode:: ipython

    In [344]: plot(t, y)
    Out[344]: [<matplotlib.lines.Line2D object at 0xcffaa6c>]
    In [345]: clf()
    In [346]: semilogy(t, y)
    Out[346]: [<matplotlib.lines.Line2D object at 0xd0eebec>]

Other example: damped spring-mass oscillator (2nd order oscillator)

The position of a mass attached to a spring obeys the 2nd order ODE ``y''
+ nu y' + om^2 y = 0``, that can be transformed in a system of two
first-order equations for the vector ``Y=(y, y')``. ::

    >>> def rhs(y,t, om, nu):
    ...     return (y[1], -om**2*y[0] - nu*y[1])
    ... 
    >>> y = odeint(rhs, (1, 0), t, args=(3, 1)) # args correspond à (om, nu)

.. image:: spring.png
   :align: center

.. sourcecode:: ipython

    In [385]: plot(t, y[:,0], lw=2, label="y")
    Out[385]: [<matplotlib.lines.Line2D object at 0xd675e2c>]
    In [386]: plot(t, y[:,1], lw=2, label="y'")
    Out[386]: [<matplotlib.lines.Line2D object at 0xd67908c>]
    In [387]: legend()
    Out[387]: <matplotlib.legend.Legend object at 0xd67e26c>


.. topic:: EDP

    .. image:: cahn.png

    .. image:: dendrite.png

    There is no EDP solver in scipy. Some EDP packages are written in
    Python, such as **fipy** (http://www.ctcms.nist.gov/fipy/) or **SfePy**
    (http://code.google.com/p/sfepy/).

Optimization and fits : ``scipy.optimize``
-----------------------------------------

**Exemple : random walker simulation (cont'd)**

.. image:: random_walk.png
   :align: center 

Let us go back to the random walker example ::

    >>> nreal = 1000 # realizations number of the walk
    >>> tmax = 200 # number of time steps for the walk
    >>> # 1 and -1 steps are drawn at random
    >>> walk = 2 * ( np.random.random_integers(0, 1, (nreal,200)) - 0.5 )
    >>> np.unique(walk) # Check that all steps have 1 or -1 values.
    array([-1.,  1.])
    >>> # The position of the walker is obtained by summing the steps
    >>> cumwalk = np.cumsum(walk, axis=1) # axis = 1 : time dimension
    >>> sq_distance = cumwalk**2
    >>> # Now let us mean over realizations of the walk 
    >>> mean_distance = np.sqrt(np.mean(sq_distance, axis=0)) 

We now fit the ``mean_distance`` array by the square root function.

.. sourcecode:: ipython

    In [40]: plot(mean_distance)
    In [41]: t = np.arange(tmax)
    In [42]: def f(A, y, x): 
       ....:	 """function to optimize"""
       ....:     err = y - A*np.sqrt(x)
       ....:     return err
       ....: 
    In [43]: coeff = scipy.optimize.leastsq(f, 0.8, args=(mean_distance, t))
    In [44]: coeff
    In [45]: coeff
    Out[45]: (1.0017300505346165, 3)
    In [46]: plot(t, coeff[0]*np.sqrt(t), lw=2)

.. image:: diffusion.png
   :align: center

A linear fit using ``np.polyfit`` of the squared distance also gives a
good estimation,

.. sourcecode:: ipython

    In [47]: t = np.arange(tmax)
    In [48]: np.polyfit(t, mean_distance**2, 1)
    Out[48]: array([ 1.00452036, -0.08389612])

however, this other fit solves a different optimization problem.

Image processing: ``scipy.ndimage``
-----------------------------------------

This submodule offers image processing routines for n-dimensional arrays
(the routines can be used on arrays with a different number of dimensions, 2-D, 3-D, etc.).

Example ::

    >>> import scipy.ndimage
    >>> lena = scipy.lena()
    >>> lena_floue = scipy.ndimage.gaussian_filter(lena, 3)
    >>> lena_rotated = scipy.ndimage.rotate(lena, 45)

.. image:: lena_ndimage.png
   :align: center
