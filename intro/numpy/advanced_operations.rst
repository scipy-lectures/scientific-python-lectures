.. For doctests
   >>> import numpy as np
   >>> from matplotlib import pyplot as plt


.. currentmodule:: numpy

Advanced operations
===================

.. contents:: Section contents
    :local:
    :depth: 1

Polynomials
-----------

Numpy also contains polynomials in different bases:

For example, :math:`3x^2 + 2x - 1`::

    >>> p = np.poly1d([3, 2, -1])
    >>> p(0)
    -1
    >>> p.roots
    array([-1.        ,  0.33333333])
    >>> p.order
    2

::

    >>> x = np.linspace(0, 1, 20)
    >>> y = np.cos(x) + 0.3*np.random.rand(20)
    >>> p = np.poly1d(np.polyfit(x, y, 3))

    >>> t = np.linspace(0, 1, 200)
    >>> plt.plot(x, y, 'o', t, p(t), '-')   # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]

.. plot:: pyplots/numpy_intro_9.py

See http://docs.scipy.org/doc/numpy/reference/routines.polynomials.poly1d.html
for more.

More polynomials (with more bases)
...................................

Numpy also has a more sophisticated polynomial interface, which supports
e.g. the Chebyshev basis.

:math:`3x^2 + 2x - 1`::

    >>> p = np.polynomial.Polynomial([-1, 2, 3]) # coefs in different order!
    >>> p(0)
    -1.0
    >>> p.roots()
    array([-1.        ,  0.33333333])
    >>> p.degree()  # In general polynomials do not always expose 'order'
    2

Example using polynomials in Chebyshev basis, for polynomials in
range ``[-1, 1]``::

    >>> x = np.linspace(-1, 1, 2000)
    >>> y = np.cos(x) + 0.3*np.random.rand(2000)
    >>> p = np.polynomial.Chebyshev.fit(x, y, 90)

    >>> t = np.linspace(-1, 1, 200)
    >>> plt.plot(x, y, 'r.')   # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(t, p(t), 'k-', lw=3)   # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]

.. plot:: pyplots/numpy_intro_10.py

The Chebyshev polynomials have some advantages in interpolation.

Loading data files
-------------------

Text files
...........

Example: :download:`populations.txt <../../data/populations.txt>`:

.. include:: ../../data/populations.txt
    :end-line: 5
    :literal:

::

    >>> data = np.loadtxt('data/populations.txt')
    >>> data    # doctest: +ELLIPSIS
    array([[  1900.,  30000.,   4000.,  48300.],
           [  1901.,  47200.,   6100.,  48200.],
           [  1902.,  70200.,   9800.,  41500.],
    ...

::

    >>> np.savetxt('pop2.txt', data)
    >>> data2 = np.loadtxt('pop2.txt')

.. note:: If you have a complicated text file, what you can try are:

   - ``np.genfromtxt``

   - Using Python's I/O functions and e.g. regexps for parsing
     (Python is quite well suited for this)

.. topic:: Reminder: Navigating the filesystem with IPython

   .. sourcecode:: ipython

       In [1]: pwd      # show current directory
       '/home/user/stuff/2011-numpy-tutorial'
       In [2]: cd ex
       '/home/user/stuff/2011-numpy-tutorial/ex'
       In [3]: ls
       populations.txt	species.txt

Images
.......

Using Matplotlib::

    >>> img = plt.imread('data/elephant.png')
    >>> img.shape, img.dtype
    ((200, 300, 3), dtype('float32'))
    >>> plt.imshow(img)     # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>
    >>> plt.savefig('plot.png')

    >>> plt.imsave('red_elephant', img[:,:,0], cmap=plt.cm.gray)

This saved only one channel (of RGB)::

    >>> plt.imshow(plt.imread('red_elephant.png'))  # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>

Other libraries::

    >>> from scipy.misc import imsave
    >>> imsave('tiny_elephant.png', img[::6,::6])
    >>> plt.imshow(plt.imread('tiny_elephant.png'), interpolation='nearest')  # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>

.. plot:: pyplots/numpy_intro_3.py


Numpy's own format
...................

Numpy has its own binary format, not portable but with efficient I/O::

    >>> data = np.ones((3, 3))
    >>> np.save('pop.npy', data)
    >>> data3 = np.load('pop.npy')

Well-known (& more obscure) file formats
.........................................

* HDF5: `h5py <http://code.google.com/p/h5py/>`__, `PyTables <http://pytables.org>`__
* NetCDF: ``scipy.io.netcdf_file``, `netcdf4-python <http://code.google.com/p/netcdf4-python/>`__, ...
* Matlab: ``scipy.io.loadmat``, ``scipy.io.savemat``
* MatrixMarket: ``scipy.io.mmread``, ``scipy.io.mmread``

... if somebody uses it, there's probably also a Python library for it.


.. topic:: Exercise: Text data files
   :class: green

   Write a Python script that loads data from :download:`populations.txt
   <../../data/populations.txt>`:: and drop the last column and the first
   5 rows. Save the smaller dataset to ``pop2.txt``.


.. loadtxt, savez, load, fromfile, tofile

.. real life: point to HDF5, NetCDF, etc.

.. EXE: use loadtxt to load a data file
.. EXE: use savez and load to save data in binary format
.. EXE: use tofile and fromfile to put and get binary data bytes in/from a file
   follow-up: .view()
.. EXE: parsing text files -- Python can do this reasonably well natively!
   throw in the mix some random text file to be parsed (eg. PPM)
.. EXE: advanced: read the data in a PPM file


.. topic:: Numpy internals

    If you are interested in the Numpy internals, there is a good discussion in
    :ref:`advanced_numpy`.

