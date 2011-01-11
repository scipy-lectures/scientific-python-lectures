Dictionary of Keys Format (DOK)
===============================

* subclass of Python dict
    * keys are `(row, column)` index tuples (no duplicate entries allowed)
    * values are corresponding non-zero values
* efficient for constructing sparse matrices incrementally
* constructor accepts:
    * dense matrix (array)
    * sparse matrix
    * shape tuple (create empty matrix)
* efficient O(1) access to individual elements
* flexible slicing, changing sparsity structure is efficient
* can be efficiently converted to a coo_matrix once constructed
* slow arithmetics (`for` loops with `dict.iteritems()`)
* use:
    * when sparsity pattern is not known apriori or changes

Examples
--------

* create a DOK matrix element by element::

    >>> mtx = sps.dok_matrix((5, 5), dtype=np.float64)
    >>> mtx
    <5x5 sparse matrix of type '<type 'numpy.float64'>'
            with 0 stored elements in Dictionary Of Keys format>
    >>> for ir in range(5):
    >>>     for ic in range(5):
    >>>         mtx[ir, ic] = 1.0 * (ir != ic)
    >>> mtx
    <5x5 sparse matrix of type '<type 'numpy.float64'>'
            with 25 stored elements in Dictionary Of Keys format>
    >>> mtx.todense()
    matrix([[ 0.,  1.,  1.,  1.,  1.],
            [ 1.,  0.,  1.,  1.,  1.],
            [ 1.,  1.,  0.,  1.,  1.],
            [ 1.,  1.,  1.,  0.,  1.],
            [ 1.,  1.,  1.,  1.,  0.]])

* something is wrong, let's start over::

    >>> mtx = sps.dok_matrix((5, 5), dtype=np.float64)
    >>> for ir in range(5):
    >>>     for ic in range(5):
    >>>         if (ir != ic): mtx[ir, ic] = 1.0
    >>> mtx
    <5x5 sparse matrix of type '<type 'numpy.float64'>'
            with 20 stored elements in Dictionary Of Keys format>

* slicing and indexing::

    >>> mtx[1, 1]
    0.0
    >>> mtx[1, 1:3]
    <1x2 sparse matrix of type '<type 'numpy.float64'>'
          with 1 stored elements in Dictionary Of Keys format>
    >>> mtx[1, 1:3].todense()
    matrix([[ 0.,  1.]])
    >>> mtx[[2,1], 1:3].todense()
    ------------------------------------------------------------
    Traceback (most recent call last):
    <snip>
    NotImplementedError: fancy indexing supported over one axis only

