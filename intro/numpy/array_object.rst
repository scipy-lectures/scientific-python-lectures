..
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")


.. currentmodule:: numpy

The NumPy array object
======================

.. contents:: Section contents
    :local:
    :depth: 1

What are NumPy and NumPy arrays?
--------------------------------

NumPy arrays
............

:**Python** objects:

    - high-level number objects: integers, floating point

    - containers: lists (costless insertion and append), dictionaries
      (fast lookup)

:**NumPy** provides:

    - extension package to Python for multi-dimensional arrays

    - closer to hardware (efficiency)

    - designed for scientific computation (convenience)

    - Also known as *array oriented computing*

|

.. sourcecode:: pycon

    >>> import numpy as np
    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])

.. tip::

    For example, An array containing:

    * values of an experiment/simulation at discrete time steps

    * signal recorded by a measurement device, e.g. sound wave

    * pixels of an image, grey-level or colour

    * 3-D data measured at different X-Y-Z positions, e.g. MRI scan

    * ...

**Why it is useful:** Memory-efficient container that provides fast numerical
operations.

.. sourcecode:: ipython

    In [1]: L = range(1000)

    In [2]: %timeit [i**2 for i in L]
    1000 loops, best of 3: 403 us per loop

    In [3]: a = np.arange(1000)

    In [4]: %timeit a**2
    100000 loops, best of 3: 12.7 us per loop


.. extension package to Python to support multidimensional arrays

.. diagram, import conventions

.. scope of this tutorial: drill in features of array manipulation in
   Python, and try to give some indication on how to get things done
   in good style

.. a fixed number of elements (cf. certain exceptions)
.. each element of same size and type
.. efficiency vs. Python lists

NumPy Reference documentation
..............................

- On the web: http://docs.scipy.org/

- Interactive help:

  .. sourcecode:: ipython

     In [5]: np.array?
     String Form:<built-in function array>
     Docstring:
     array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0, ...

  .. tip:

   .. sourcecode:: pycon

     >>> help(np.array) # doctest: +ELLIPSIS
     Help on built-in function array in module numpy.core.multiarray:
     <BLANKLINE>
     array(...)
         array(object, dtype=None, copy=True, order=None, subok=False, ...


- Looking for something:

  .. sourcecode:: pycon

     >>> np.lookfor('create array') # doctest: +SKIP
     Search results for 'create array'
     ---------------------------------
     numpy.array
         Create an array.
     numpy.memmap
         Create a memory-map to an array stored in a *binary* file on disk.

  .. sourcecode:: ipython

     In [6]: np.con*?
     np.concatenate
     np.conj
     np.conjugate
     np.convolve

Import conventions
..................

The recommended convention to import numpy is:

.. sourcecode:: pycon

   >>> import numpy as np


Creating arrays
---------------

Manual construction of arrays
..............................

* **1-D**:

  .. sourcecode:: pycon

    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])
    >>> a.ndim
    1
    >>> a.shape
    (4,)
    >>> len(a)
    4

* **2-D, 3-D, ...**:

  .. sourcecode:: pycon

    >>> b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
    >>> b
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> b.ndim
    2
    >>> b.shape
    (2, 3)
    >>> len(b)     # returns the size of the first dimension
    2

    >>> c = np.array([[[1], [2]], [[3], [4]]])
    >>> c
    array([[[1],
            [2]],
    <BLANKLINE>
           [[3],
            [4]]])
    >>> c.shape
    (2, 2, 1)

.. topic:: **Exercise: Simple arrays**
    :class: green

    * Create a simple two dimensional array. First, redo the examples
      from above. And then create your own: how about odd numbers
      counting backwards on the first row, and even numbers on the second?
    * Use the functions :func:`len`, :func:`numpy.shape` on these arrays.
      How do they relate to each other? And to the ``ndim`` attribute of
      the arrays?

Functions for creating arrays
..............................

.. tip::

    In practice, we rarely enter items one by one...

* Evenly spaced:

  .. sourcecode:: pycon

    >>> a = np.arange(10) # 0 .. n-1  (!)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = np.arange(1, 9, 2) # start, end (exclusive), step
    >>> b
    array([1, 3, 5, 7])

* or by number of points:

  .. sourcecode:: pycon

    >>> c = np.linspace(0, 1, 6)   # start, end, num-points
    >>> c
    array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
    >>> d = np.linspace(0, 1, 5, endpoint=False)
    >>> d
    array([ 0. ,  0.2,  0.4,  0.6,  0.8])

* Common arrays:

  .. sourcecode:: pycon

    >>> a = np.ones((3, 3))  # reminder: (3, 3) is a tuple
    >>> a
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    >>> b = np.zeros((2, 2))
    >>> b
    array([[ 0.,  0.],
           [ 0.,  0.]])
    >>> c = np.eye(3)
    >>> c
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> d = np.diag(np.array([1, 2, 3, 4]))
    >>> d
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

* :mod:`np.random`: random numbers (Mersenne Twister PRNG):

  .. sourcecode:: pycon

    >>> a = np.random.rand(4)       # uniform in [0, 1]
    >>> a  # doctest: +SKIP
    array([ 0.95799151,  0.14222247,  0.08777354,  0.51887998])

    >>> b = np.random.randn(4)      # Gaussian
    >>> b  # doctest: +SKIP
    array([ 0.37544699, -0.11425369, -0.47616538,  1.79664113])

    >>> np.random.seed(1234)        # Setting the random seed

.. topic:: **Exercise: Creating arrays using functions**
   :class: green

   * Experiment with ``arange``, ``linspace``, ``ones``, ``zeros``, ``eye`` and
     ``diag``.
   * Create different kinds of arrays with random numbers.
   * Try setting the seed before creating an array with random values.
   * Look at the function ``np.empty``. What does it do? When might this be
     useful?

.. EXE: construct 1 2 3 4 5
.. EXE: construct -5, -4, -3, -2, -1
.. EXE: construct 2 4 6 8
.. EXE: look what is in an empty() array
.. EXE: construct 15 equispaced numbers in range [0, 10]

Basic data types
----------------

You may have noticed that, in some instances, array elements are displayed with
a trailing dot (e.g. ``2.`` vs ``2``). This is due to a difference in the
data-type used:

.. sourcecode:: pycon

    >>> a = np.array([1, 2, 3])
    >>> a.dtype
    dtype('int64')

    >>> b = np.array([1., 2., 3.])
    >>> b.dtype
    dtype('float64')

.. tip::

    Different data-types allow us to store data more compactly in memory,
    but most of the time we simply work with floating point numbers.
    Note that, in the example above, NumPy auto-detects the data-type
    from the input.

-----------------------------

You can explicitly specify which data-type you want:

.. sourcecode:: pycon

    >>> c = np.array([1, 2, 3], dtype=float)
    >>> c.dtype
    dtype('float64')


The **default** data type is floating point:

.. sourcecode:: pycon

    >>> a = np.ones((3, 3))
    >>> a.dtype
    dtype('float64')

There are also other types:

:Complex:

  .. sourcecode:: pycon

        >>> d = np.array([1+2j, 3+4j, 5+6*1j])
        >>> d.dtype
        dtype('complex128')

:Bool:

  .. sourcecode:: pycon

        >>> e = np.array([True, False, False, True])
        >>> e.dtype
        dtype('bool')

:Strings:

  .. sourcecode:: pycon

        >>> f = np.array(['Bonjour', 'Hello', 'Hallo',])
        >>> f.dtype     # <--- strings containing max. 7 letters  # doctest: +SKIP
        dtype('S7')

:Much more:

    * ``int32``
    * ``int64``
    * ``uint32``
    * ``uint64``

.. XXX: mention: astype


Basic visualization
-------------------

Now that we have our first data arrays, we are going to visualize them.

Start by launching IPython:

.. sourcecode:: bash

    $ ipython

Or the notebook:

.. sourcecode:: bash

   $ ipython notebook

Once IPython has started, enable interactive plots:

.. sourcecode:: pycon

    >>> %matplotlib  # doctest: +SKIP

Or, from the notebook, enable plots in the notebook:

.. sourcecode:: pycon

    >>> %matplotlib inline # doctest: +SKIP

The ``inline`` is important for the notebook, so that plots are displayed in
the notebook and not in a new window.

*Matplotlib* is a 2D plotting package. We can import its functions as below:

.. sourcecode:: pycon

    >>> import matplotlib.pyplot as plt  # the tidy way

And then use (note that you have to use ``show`` explicitly if you have not enabled interactive plots with ``%matplotlib``):

.. sourcecode:: pycon

    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP
    >>> plt.show()           # <-- shows the plot (not needed with interactive plots) # doctest: +SKIP

Or, if you have enabled interactive plots with ``%matplotlib``:

.. sourcecode:: pycon

    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP

* **1D plotting**:

  .. sourcecode:: pycon

    >>> x = np.linspace(0, 3, 20)
    >>> y = np.linspace(0, 9, 20)
    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(x, y, 'o')  # dot plot    # doctest: +SKIP
    [<matplotlib.lines.Line2D object at ...>]

  .. plot:: pyplots/numpy_intro_1.py

* **2D arrays** (such as images):

  .. sourcecode:: pycon

    >>> image = np.random.rand(30, 30)
    >>> plt.imshow(image, cmap=plt.cm.hot)    # doctest: +SKIP
    >>> plt.colorbar()    # doctest: +SKIP
    <matplotlib.colorbar.Colorbar instance at ...>

  .. plot:: pyplots/numpy_intro_2.py

.. seealso:: More in the: :ref:`matplotlib chapter <matplotlib>`

.. topic:: **Exercise: Simple visualizations**
   :class: green

   * Plot some simple arrays: a cosine as a function of time and a 2D
     matrix.
   * Try using the ``gray`` colormap on the 2D matrix.

.. * **3D plotting**:
..
..   For 3D visualization, we can use another package: **Mayavi**. A quick example:
..   start by **relaunching iPython** with these options: **ipython --pylab=wx**
..   (or **ipython -pylab -wthread** in IPython < 0.10).
..
..   .. image:: surf.png
..      :align: right
..      :scale: 60
..
..   .. sourcecode:: ipython
..
..       In [58]: from mayavi import mlab
..       In [61]: mlab.surf(image)
..       Out[61]: <enthought.mayavi.modules.surface.Surface object at ...>
..       In [62]: mlab.axes()
..       Out[62]: <enthought.mayavi.modules.axes.Axes object at ...>
..
..   .. tip::
..
..    The mayavi/mlab window that opens is interactive: by clicking on the
..    left mouse button you can rotate the image, zoom with the mouse wheel,
..    etc.
..
..    For more information on Mayavi :
..    https://github.enthought.com/mayavi/mayavi
..
..   .. seealso:: More in the :ref:`Mayavi chapter <mayavi-label>`


Indexing and slicing
--------------------

The items of an array can be accessed and assigned to the same way as
other Python sequences (e.g. lists):

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> a[0], a[2], a[-1]
    (0, 2, 9)

.. warning::

   Indices begin at 0, like other Python sequences (and C/C++).
   In contrast, in Fortran or Matlab, indices begin at 1.

The usual python idiom for reversing a sequence is supported:

.. sourcecode:: pycon

   >>> a[::-1]
   array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

For multidimensional arrays, indexes are tuples of integers:

.. sourcecode:: pycon

    >>> a = np.diag(np.arange(3))
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 2]])
    >>> a[1, 1]
    1
    >>> a[2, 1] = 10 # third line, second column
    >>> a
    array([[ 0,  0,  0],
           [ 0,  1,  0],
           [ 0, 10,  2]])
    >>> a[1]
    array([0, 1, 0])


.. note::

  * In 2D, the first dimension corresponds to **rows**, the second 
    to **columns**.
  * for multidimensional ``a``, ``a[0]`` is interpreted by
    taking all elements in the unspecified dimensions.

**Slicing**: Arrays, like other Python sequences can also be sliced:

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> a[2:9:3] # [start:end:step]
    array([2, 5, 8])

Note that the last index is not included! :

.. sourcecode:: pycon

    >>> a[:4]
    array([0, 1, 2, 3])

All three slice components are not required: by default, `start` is 0,
`end` is the last and `step` is 1:

.. sourcecode:: pycon

    >>> a[1:3]
    array([1, 2])
    >>> a[::2]
    array([0, 2, 4, 6, 8])
    >>> a[3:]
    array([3, 4, 5, 6, 7, 8, 9])

A small illustrated summary of NumPy indexing and slicing...

.. only:: latex

    .. image:: images/numpy_indexing.png
        :align: center

.. only:: html

    .. image:: images/numpy_indexing.png
        :align: center
        :width: 70%

You can also combine assignment and slicing:

.. sourcecode:: pycon

   >>> a = np.arange(10)
   >>> a[5:] = 10
   >>> a
   array([ 0,  1,  2,  3,  4, 10, 10, 10, 10, 10])
   >>> b = np.arange(5)
   >>> a[5:] = b[::-1]
   >>> a
   array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])

.. topic:: **Exercise: Indexing and slicing**
   :class: green

   * Try the different flavours of slicing, using ``start``, ``end`` and
     ``step``: starting from a linspace, try to obtain odd numbers
     counting backwards, and even numbers counting forwards.
   * Reproduce the slices in the diagram above. You may
     use the following expression to create the array:

     .. sourcecode:: pycon

        >>> np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
        array([[ 0,  1,  2,  3,  4,  5],
               [10, 11, 12, 13, 14, 15],
               [20, 21, 22, 23, 24, 25],
               [30, 31, 32, 33, 34, 35],
               [40, 41, 42, 43, 44, 45],
               [50, 51, 52, 53, 54, 55]])

.. topic:: **Exercise: Array creation**
    :class: green

    Create the following arrays (with correct data types)::

        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 2],
         [1, 6, 1, 1]]

        [[0., 0., 0., 0., 0.],
         [2., 0., 0., 0., 0.],
         [0., 3., 0., 0., 0.],
         [0., 0., 4., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 6.]]

    Par on course: 3 statements for each

    *Hint*: Individual array elements can be accessed similarly to a list,
    e.g. ``a[1]`` or ``a[1, 2]``.

    *Hint*: Examine the docstring for ``diag``.

.. topic:: Exercise: Tiling for array creation
    :class: green

    Skim through the documentation for ``np.tile``, and use this function
    to construct the array::

        [[4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1],
         [4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1]]

Copies and views
----------------

A slicing operation creates a **view** on the original array, which is
just a way of accessing array data. Thus the original array is not
copied in memory. You can use ``np.may_share_memory()`` to check if two arrays
share the same memory block. Note however, that this uses heuristics and may
give you false positives.

**When modifying the view, the original array is modified as well**:

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = a[::2]
    >>> b
    array([0, 2, 4, 6, 8])
    >>> np.may_share_memory(a, b)
    True
    >>> b[0] = 12
    >>> b
    array([12,  2,  4,  6,  8])
    >>> a   # (!)
    array([12,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    >>> a = np.arange(10)
    >>> c = a[::2].copy()  # force a copy
    >>> c[0] = 12
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> np.may_share_memory(a, c)
    False



This behavior can be surprising at first sight... but it allows to save both
memory and time.


.. EXE: [1, 2, 3, 4, 5] -> [1, 2, 3]
.. EXE: [1, 2, 3, 4, 5] -> [4, 5]
.. EXE: [1, 2, 3, 4, 5] -> [1, 3, 5]
.. EXE: [1, 2, 3, 4, 5] -> [2, 4]
.. EXE: create an array [1, 1, 1, 1, 0, 0, 0]
.. EXE: create an array [0, 0, 0, 0, 1, 1, 1]
.. EXE: create an array [0, 1, 0, 1, 0, 1, 0]
.. EXE: create an array [1, 0, 1, 0, 1, 0, 1]
.. EXE: create an array [1, 0, 2, 0, 3, 0, 4]
.. CHA: archimedean sieve

.. topic:: Worked example: Prime number sieve
   :class: green

   .. image:: images/prime-sieve.png

   Compute prime numbers in 0--99, with a sieve

   * Construct a shape (100,) boolean array ``is_prime``,
     filled with True in the beginning:

   .. sourcecode:: pycon

        >>> is_prime = np.ones((100,), dtype=bool)

   * Cross out 0 and 1 which are not primes:

   .. sourcecode:: pycon

       >>> is_prime[:2] = 0

   * For each integer ``j`` starting from 2, cross out its higher multiples:

   .. sourcecode:: pycon

       >>> N_max = int(np.sqrt(len(is_prime) - 1))
       >>> for j in range(2, N_max + 1):
       ...     is_prime[2*j::j] = False

   * Skim through ``help(np.nonzero)``, and print the prime numbers

   * Follow-up:

     - Move the above code into a script file named ``prime_sieve.py``

     - Run it to check it works

     - Use the optimization suggested in `the sieve of Eratosthenes
       <https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes>`_:

      1. Skip ``j`` which are already known to not be primes

      2. The first number to cross out is :math:`j^2`

Fancy indexing
--------------

.. tip::

    NumPy arrays can be indexed with slices, but also with boolean or
    integer arrays (**masks**). This method is called *fancy indexing*.
    It creates **copies not views**.

Using boolean masks
...................

.. sourcecode:: pycon

    >>> np.random.seed(3)
    >>> a = np.random.random_integers(0, 20, 15)
    >>> a
    array([10,  3,  8,  0, 19, 10, 11,  9, 10,  6,  0, 20, 12,  7, 14])
    >>> (a % 3 == 0)
    array([False,  True, False,  True, False, False, False,  True, False,
            True,  True, False,  True, False, False], dtype=bool)
    >>> mask = (a % 3 == 0)
    >>> extract_from_a = a[mask] # or,  a[a%3==0]
    >>> extract_from_a           # extract a sub-array with the mask
    array([ 3,  0,  9,  6,  0, 12])

Indexing with a mask can be very useful to assign a new value to a sub-array:

.. sourcecode:: pycon

    >>> a[a % 3 == 0] = -1
    >>> a
    array([10, -1,  8, -1, 19, 10, 11, -1, 10, -1, -1, 20, -1,  7, 14])


Indexing with an array of integers
..................................

.. sourcecode:: pycon

    >>> a = np.arange(0, 100, 10)
    >>> a
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

Indexing can be done with an array of integers, where the same index is repeated
several time:

.. sourcecode:: pycon

    >>> a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list
    array([20, 30, 20, 40, 20])

New values can be assigned with this kind of indexing:

.. sourcecode:: pycon

    >>> a[[9, 7]] = -100
    >>> a
    array([   0,   10,   20,   30,   40,   50,   60, -100,   80, -100])

.. tip::

  When a new array is created by indexing with an array of integers, the
  new array has the same shape than the array of integers:

  .. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> idx = np.array([[3, 4], [9, 7]])
    >>> idx.shape
    (2, 2)
    >>> a[idx]
    array([[3, 4],
           [9, 7]])


____

The image below illustrates various fancy indexing applications

.. only:: latex

    .. image:: images/numpy_fancy_indexing.png
        :align: center

.. only:: html

    .. image:: images/numpy_fancy_indexing.png
        :align: center
        :width: 80%

.. topic:: **Exercise: Fancy indexing**
    :class: green

    * Again, reproduce the fancy indexing shown in the diagram above.
    * Use fancy indexing on the left and array creation on the right to assign
      values into an array, for instance by setting parts of the array in
      the diagram above to zero.

.. We can even use fancy indexing and :ref:`broadcasting <broadcasting>` at
.. the same time:
..
.. .. sourcecode:: pycon
..
..     >>> a = np.arange(12).reshape(3,4)
..     >>> a
..     array([[ 0,  1,  2,  3],
..            [ 4,  5,  6,  7],
..            [ 8,  9, 10, 11]])
..     >>> i = np.array([[0, 1], [1, 2]])
..     >>> a[i, 2] # same as a[i, 2*np.ones((2, 2), dtype=int)]
..     array([[ 2,  6],
..            [ 6, 10]])


