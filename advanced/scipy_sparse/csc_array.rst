.. For doctests
   >>> import numpy as np
   >>> import scipy as sp


Compressed Sparse Column Format (CSC)
=====================================

* column oriented
    * three NumPy arrays: `indices`, `indptr`, `data`
        * `indices` is array of row indices
        * `data` is array of corresponding nonzero values
        * `indptr` points to column starts in `indices` and `data`
        * length is `n_col + 1`, last item = number of values = length of both
          `indices` and `data`
        * nonzero values of the `i`-th column are `data[indptr[i]:indptr[i+1]]`
          with row indices `indices[indptr[i]:indptr[i+1]]`
        * item `(i, j)` can be accessed as `data[indptr[j]+k]`, where `k` is
          position of `i` in `indices[indptr[j]:indptr[j+1]]`
    * subclass of :class:`_cs_matrix` (common CSR/CSC functionality)
        * subclass of :class:`_data_matrix` (sparse array classes with
          `.data` attribute)
* fast matrix vector products and other arithmetic (sparsetools)
* constructor accepts:
    * dense array/matrix
    * sparse array/matrix
    * shape tuple (create empty array)
    * `(data, coords)` tuple
    * `(data, indices, indptr)` tuple
* efficient column slicing, column-oriented operations
* slow row slicing, expensive changes to the sparsity structure
* use:
    * actual computations (most linear solvers support this format)

Examples
--------

* create empty CSC array::

    >>> mtx = sp.sparse.csc_array((3, 4), dtype=np.int8)
    >>> mtx.toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

* create using `(data, coords)` tuple::

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> mtx = sp.sparse.csc_array((data, (row, col)), shape=(3, 3))
    >>> mtx
    <3x3 sparse array of type '<... 'numpy.int64'>'
            with 6 stored elements in Compressed Sparse Column format>
    >>> mtx.toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]]...)
    >>> mtx.data
    array([1, 4, 5, 2, 3, 6]...)
    >>> mtx.indices
    array([0, 2, 2, 0, 1, 2])
    >>> mtx.indptr
    array([0, 2, 3, 6])

* create using `(data, indices, indptr)` tuple::

    >>> data = np.array([1, 4, 5, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> indptr = np.array([0, 2, 3, 6])
    >>> mtx = sp.sparse.csc_array((data, indices, indptr), shape=(3, 3))
    >>> mtx.toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])
