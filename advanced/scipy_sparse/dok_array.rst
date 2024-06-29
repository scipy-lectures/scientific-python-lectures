.. For doctests
   >>> import numpy as np
   >>> import scipy as sp


Dictionary of Keys Format (DOK)
===============================

* subclass of Python dict
    * keys are `(row, column)` index tuples (no duplicate entries allowed)
    * values are corresponding non-zero values
* efficient for constructing sparse arrays incrementally
* constructor accepts:
    * dense array/matrix
    * sparse array/matrix
    * shape tuple (create empty array)
* efficient O(1) access to individual elements
* flexible slicing, changing sparsity structure is efficient
* can be efficiently converted to a coo_array once constructed
* slow arithmetic (`for` loops with `dict.items()`)
* use:
    * when sparsity pattern is not known apriori or changes

Examples
--------

* create a DOK array element by element::

    >>> mtx = sp.sparse.dok_array((5, 5), dtype=np.float64)
    >>> mtx
    <Dictionary Of Keys sparse array of dtype 'float64'
            with 0 stored elements and shape (5, 5)>
    >>> for ir in range(5):
    ...     for ic in range(5):
    ...         mtx[ir, ic] = 1.0 * (ir != ic)
    >>> mtx
    <Dictionary Of Keys sparse array of dtype 'float64'
            with 20 stored elements and shape (5, 5)>
    >>> mtx.toarray()
    array([[0.,  1.,  1.,  1.,  1.],
           [1.,  0.,  1.,  1.,  1.],
           [1.,  1.,  0.,  1.,  1.],
           [1.,  1.,  1.,  0.,  1.],
           [1.,  1.,  1.,  1.,  0.]])

* slicing and indexing::

    >>> mtx[1, 1]
    np.float64(0.0)
    >>> mtx[[1], 1:3]
    <Dictionary Of Keys sparse array of dtype 'float64'
            with 1 stored elements and shape (1, 2)>
    >>> mtx[[1], 1:3].toarray()
    array([[0.,  1.]])
    >>> mtx[[2, 1], 1:3].toarray()
    array([[1.,  0.],
           [0.,  1.]])
