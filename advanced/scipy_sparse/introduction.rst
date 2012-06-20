Introduction
============

(dense) matrix is:

* mathematical object
* data structure for storing a 2D array of values

important features:

* memory allocated once for all items
    * usually a contiguous chunk, think NumPy ndarray
* *fast* access to individual items (*)

Why Sparse Matrices?
--------------------

* the memory, that grows like `n**2`
* small example (double precision matrix)::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 1e6, 10)
    >>> plt.plot(x, 8.0 * (x**2) / 1e6, lw=5)    # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('size n')    # doctest: +ELLIPSIS
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('memory [MB]')    # doctest: +ELLIPSIS
    <matplotlib.text.Text object at ...>

Sparse Matrices vs. Sparse Matrix Storage Schemes
-------------------------------------------------

* sparse matrix is a matrix, which is *almost empty*
* storing all the zeros is wasteful -> store only nonzero items
* think **compression**
* pros: huge memory savings
* cons: depends on actual storage scheme, (*) usually does not hold

Typical Applications
--------------------

* solution of partial differential equations (PDEs)
    * the *finite element method*
    * mechanical engineering, electrotechnics, physics, ...
* graph theory
    * nonzero at `(i, j)` means that node `i` is connected to node `j`
* ...

Prerequisites
-------------

recent versions of

* ``numpy``
* ``scipy``
* ``matplotlib`` (optional)
* ``ipython`` (the enhancements come handy)

Sparsity Structure Visualization
--------------------------------

* :func:`spy` from ``matplotlib``
* example plots:

.. image:: figures/graph.png
.. image:: figures/graph_g.png
.. image:: figures/graph_rcm.png
