.. For doctests
   >>> import numpy as np
   >>> import scipy as sp


Block Compressed Row Format (BSR)
=================================

* basically a CSR with dense sub-matrices of fixed shape instead of scalar
  items
    * block size `(R, C)` must evenly divide the shape of the matrix `(M, N)`
    * three NumPy arrays: `indices`, `indptr`, `data`
        * `indices` is array of column indices for each block
        * `data` is array of corresponding nonzero values of shape `(nnz, R, C)`
        * ...
    * subclass of :class:`_cs_matrix` (common CSR/CSC functionality)
        * subclass of :class:`_data_matrix` (sparse matrix classes with
          `.data` attribute)
* fast matrix vector products and other arithmetic (sparsetools)
* constructor accepts:
    * dense array/matrix
    * sparse array/matrix
    * shape tuple (create empty array)
    * `(data, coords)` tuple
    * `(data, indices, indptr)` tuple
* many arithmetic operations considerably more efficient than CSR for
  sparse matrices with dense sub-matrices
* use:
    * like CSR
    * vector-valued finite element discretizations

Examples
--------

* create empty BSR array with (1, 1) block size (like CSR...)::

    >>> mtx = sp.sparse.bsr_array((3, 4), dtype=np.int8)
    >>> mtx
    <3x4 sparse array of type '<... 'numpy.int8'>'
            with 0 stored elements (blocksize = 1x1) in Block Sparse Row format>
    >>> mtx.toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

* create empty BSR array with (3, 2) block size::

    >>> mtx = sp.sparse.bsr_array((3, 4), blocksize=(3, 2), dtype=np.int8)
    >>> mtx
    <3x4 sparse array of type '<... 'numpy.int8'>'
            with 0 stored elements (blocksize = 3x2) in Block Sparse Row format>
    >>> mtx.toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

  * a bug?

* create using `(data, coords)` tuple with (1, 1) block size (like CSR...)::

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> mtx = sp.sparse.bsr_array((data, (row, col)), shape=(3, 3))
    >>> mtx
    <3x3 sparse array of type '<... 'numpy.int64'>'
            with 6 stored elements (blocksize = 1x1) in Block Sparse Row format>
    >>> mtx.toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]]...)
    >>> mtx.data
    array([[[1]],
    <BLANKLINE>
           [[2]],
    <BLANKLINE>
           [[3]],
    <BLANKLINE>
           [[4]],
    <BLANKLINE>
           [[5]],
    <BLANKLINE>
           [[6]]]...)
    >>> mtx.indices
    array([0, 2, 2, 0, 1, 2])
    >>> mtx.indptr
    array([0, 2, 3, 6])

* create using `(data, indices, indptr)` tuple with (2, 2) block size::

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> mtx = sp.sparse.bsr_array((data, indices, indptr), shape=(6, 6))
    >>> mtx.toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])
    >>> data
    array([[[1, 1],
            [1, 1]],
    <BLANKLINE>
           [[2, 2],
            [2, 2]],
    <BLANKLINE>
           [[3, 3],
            [3, 3]],
    <BLANKLINE>
           [[4, 4],
            [4, 4]],
    <BLANKLINE>
           [[5, 5],
            [5, 5]],
    <BLANKLINE>
           [[6, 6],
            [6, 6]]])
