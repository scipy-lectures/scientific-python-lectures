.. for doctests
    >>> import numpy as np
    >>> import scipy as sp


Diagonal Format (DIA)
=====================

* very simple scheme
* diagonals in dense NumPy array of shape `(n_diag, length)`
    * fixed length -> waste space a bit when far from main diagonal
    * subclass of :class:`_data_matrix` (sparse array classes with
      `.data` attribute)
* offset for each diagonal
    * 0 is the main diagonal
    * negative offset = below
    * positive offset = above
* fast matrix * vector (sparsetools)
* fast and easy item-wise operations
    * manipulate data array directly (fast NumPy machinery)
* constructor accepts:
    * dense array/matrix
    * sparse array/matrix
    * shape tuple (create empty array)
    * `(data, offsets)` tuple
* no slicing, no individual item access
* use:
    * rather specialized
    * solving PDEs by finite differences
    * with an iterative solver

Examples
--------

* create some DIA arrays::

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> data
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    >>> offsets = np.array([0, -1, 2])
    >>> mtx = sp.sparse.dia_array((data, offsets), shape=(4, 4))
    >>> mtx
    <4x4 sparse array of type '<... 'numpy.int64'>'
            with 9 stored elements (3 diagonals) in DIAgonal format>
    >>> mtx.toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> data = np.arange(12).reshape((3, 4)) + 1
    >>> data
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> mtx = sp.sparse.dia_array((data, offsets), shape=(4, 4))
    >>> mtx.data
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> mtx.offsets
    array([ 0, -1,  2], dtype=int32)
    >>> print(mtx)
      (0, 0)        1
      (1, 1)        2
      (2, 2)        3
      (3, 3)        4
      (1, 0)        5
      (2, 1)        6
      (3, 2)        7
      (0, 2)        11
      (1, 3)        12
    >>> mtx.toarray()
    array([[ 1,  0, 11,  0],
           [ 5,  2,  0, 12],
           [ 0,  6,  3,  0],
           [ 0,  0,  7,  4]])

* explanation with a scheme::

    offset: row

         2:  9
         1:  --10------
         0:  1  . 11  .
        -1:  5  2  . 12
        -2:  .  6  3  .
        -3:  .  .  7  4
             ---------8

* matrix-vector multiplication

    >>> vec = np.ones((4, ))
    >>> vec
    array([1.,  1.,  1.,  1.])
    >>> mtx @ vec
    array([12.,  19.,   9.,  11.])
    >>> (mtx * vec).toarray()
    array([[ 1.,   0.,  11.,   0.],
           [ 5.,   2.,   0.,  12.],
           [ 0.,   6.,   3.,   0.],
           [ 0.,   0.,   7.,   4.]])
