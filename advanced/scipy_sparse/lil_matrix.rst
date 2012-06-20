..   
   >>> import numpy as np
   >>> np.random.seed(0)
   >>> from scipy import sparse

List of Lists Format (LIL)
==========================

* row-based linked list
    * each row is a Python list (sorted) of column indices of non-zero elements
    * rows stored in a NumPy array (`dtype=np.object`)
    * non-zero values data stored analogously
* efficient for constructing sparse matrices incrementally
* constructor accepts:
    * dense matrix (array)
    * sparse matrix
    * shape tuple (create empty matrix)
* flexible slicing, changing sparsity structure is efficient
* slow arithmetics, slow column slicing due to being row-based
* use:
    * when sparsity pattern is not known apriori or changes
    * example: reading a sparse matrix from a text file

Examples
--------

* create an empty LIL matrix::

    >>> mtx = sparse.lil_matrix((4, 5))

* prepare random data::

    >>> from numpy.random import rand
    >>> data = np.round(rand(2, 3))
    >>> data
    array([[ 1.,  1.,  1.],
           [ 1.,  0.,  1.]])

* assign the data using fancy indexing::

    >>> mtx[:2, [1, 2, 3]] = data
    >>> mtx  # doctest: +NORMALIZE_WHITESPACE
    <4x5 sparse matrix of type '<type 'numpy.float64'>'
            with 5 stored elements in LInked List format>
    >>> print mtx   # doctest: +NORMALIZE_WHITESPACE
      (0, 1)  1.0
      (0, 2)  1.0
      (0, 3)  1.0
      (1, 1)  1.0
      (1, 3)  1.0
    >>> mtx.todense()
    matrix([[ 0.,  1.,  1.,  1.,  0.],
            [ 0.,  1.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]])
    >>> mtx.toarray()
    array([[ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

* more slicing and indexing::

    >>> mtx = sparse.lil_matrix([[0, 1, 2, 0], [3, 0, 1, 0], [1, 0, 0, 1]])
    >>> mtx.todense()
    matrix([[0, 1, 2, 0],
            [3, 0, 1, 0],
            [1, 0, 0, 1]])
    >>> print mtx # doctest: +NORMALIZE_WHITESPACE
      (0, 1)    1
      (0, 2)    2
      (1, 0)    3
      (1, 2)    1
      (2, 0)    1
      (2, 3)    1
    >>> mtx[:2, :] # doctest: +NORMALIZE_WHITESPACE
    <2x4 sparse matrix of type '<type 'numpy.int64'>'
      with 4 stored elements in LInked List format>
    >>> mtx[:2, :].todense()
    matrix([[0, 1, 2, 0],
            [3, 0, 1, 0]])
    >>> mtx[1:2, [0,2]].todense()
    matrix([[3, 1]])
    >>> mtx.todense()
    matrix([[0, 1, 2, 0],
            [3, 0, 1, 0],
            [1, 0, 0, 1]])
