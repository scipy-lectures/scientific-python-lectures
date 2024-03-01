..
   >>> import numpy as np
   >>> import scipy as sp

List of Lists Format (LIL)
==========================

* row-based linked list
    * each row is a Python list (sorted) of column indices of non-zero elements
    * rows stored in a NumPy array (`dtype=np.object`)
    * non-zero values data stored analogously
* efficient for constructing sparse arrays incrementally
* constructor accepts:
    * dense array/matrix
    * sparse array/matrix
    * shape tuple (create empty array)
* flexible slicing, changing sparsity structure is efficient
* slow arithmetic, slow column slicing due to being row-based
* use:
    * when sparsity pattern is not known apriori or changes
    * example: reading a sparse array from a text file

Examples
--------

* create an empty LIL array::

    >>> mtx = sp.sparse.lil_array((4, 5))

* prepare random data::

    >>> rng = np.random.default_rng(27446968)
    >>> data = np.round(rng.random((2, 3)))
    >>> data
    array([[1.,  0.,  1.],
           [0.,  0.,  1.]])

* assign the data using fancy indexing::

    >>> mtx[:2, [1, 2, 3]] = data
    >>> mtx
    <4x5 sparse array of type '<... 'numpy.float64'>'
            with 3 stored elements in List of Lists format>
    >>> print(mtx)
      (0, 1)    1.0
      (0, 3)    1.0
      (1, 3)    1.0
    >>> mtx.toarray()
    array([[0., 1., 0., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> mtx.toarray()
    array([[0., 1., 0., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

* more slicing and indexing::

    >>> mtx = sp.sparse.lil_array([[0, 1, 2, 0], [3, 0, 1, 0], [1, 0, 0, 1]])
    >>> mtx.toarray()
    array([[0, 1, 2, 0],
           [3, 0, 1, 0],
           [1, 0, 0, 1]]...)
    >>> print(mtx)
      (0, 1)    1
      (0, 2)    2
      (1, 0)    3
      (1, 2)    1
      (2, 0)    1
      (2, 3)    1
    >>> mtx[:2, :]
    <2x4 sparse array of type '<... 'numpy.int64'>'
      with 4 stored elements in List of Lists format>
    >>> mtx[:2, :].toarray()
    array([[0, 1, 2, 0],
           [3, 0, 1, 0]]...)
    >>> mtx[1:2, [0,2]].toarray()
    array([[3, 1]]...)
    >>> mtx.toarray()
    array([[0, 1, 2, 0],
           [3, 0, 1, 0],
           [1, 0, 0, 1]]...)
