.. for doctests
   >>> import numpy as np
   >>> import scipy as sp


Coordinate Format (COO)
=======================

* also known as the 'ijv' or 'triplet' format
    * three NumPy arrays: `row`, `col`, `data`.
    * attribute `coords` is the tuple `(row, col)`
    * `data[i]` is value at `(row[i], col[i])` position
    * permits duplicate entries
    * subclass of :class:`_data_matrix` (sparse matrix classes with
      `.data` attribute)
* fast format for constructing sparse arrays
* constructor accepts:
    * dense array/matrix
    * sparse array/matrix
    * shape tuple (create empty matrix)
    * `(data, coords)` tuple
* very fast conversion to and from CSR/CSC formats
* fast matrix * vector (sparsetools)
* fast and easy item-wise operations
    * manipulate data array directly (fast NumPy machinery)
* no slicing, no arithmetic (directly, converts to CSR)
* use:
    * facilitates fast conversion among sparse formats
    * when converting to other format (usually CSR or CSC), duplicate
      entries are summed together

        * facilitates efficient construction of finite element matrices

Examples
--------

* create empty COO array::

    >>> mtx = sp.sparse.coo_array((3, 4), dtype=np.int8)
    >>> mtx.toarray()
    array([[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], dtype=int8)

* create using `(data, ij)` tuple::

    >>> row = np.array([0, 3, 1, 0])
    >>> col = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> mtx = sp.sparse.coo_array((data, (row, col)), shape=(4, 4))
    >>> mtx
    <4x4 sparse array of type '<... 'numpy.int64'>'
            with 4 stored elements in COOrdinate format>
    >>> mtx.toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

* duplicates entries are summed together::

    >>> row = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> mtx = sp.sparse.coo_array((data, (row, col)), shape=(4, 4))
    >>> mtx.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

* no slicing...::

    >>> mtx[2, 3]
    Traceback (most recent call last):
    ...
    TypeError: 'coo_array' object ...
