Storage Schemes
===============

* seven sparse array types in scipy.sparse:
    1. csr_array: Compressed Sparse Row format
    2. csc_array: Compressed Sparse Column format
    3. bsr_array: Block Sparse Row format
    4. lil_array: List of Lists format
    5. dok_array: Dictionary of Keys format
    6. coo_array: COOrdinate format (aka IJV, triplet format)
    7. dia_array: DIAgonal format
* each suitable for some tasks
* many employ sparsetools C++ module by Nathan Bell
* assume the following is imported::

  >>> import numpy as np
  >>> import scipy as sp
  >>> import matplotlib.pyplot as plt


* **warning** for Numpy users:
    * passing a sparse array object to NumPy functions that expect
      ndarray/matrix does not work. Use sparse functions.
    * the older csr_matrix classes use '*' for matrix multiplication (dot product)
      and 'A.multiply(B)' for elementwise multiplication.
    * the newer csr_array uses '@' for dot product and '*' for elementwise multiplication
    * sparse arrays can be 1D or 2D, but not nD for n > 2 (unlike Numpy arrays).

Common Methods
--------------

* all scipy.sparse array classes are subclasses of :class:`sparray`
    * default implementation of arithmetic operations
        * always converts to CSR
        * subclasses override for efficiency
    * shape, data type, set/get
    * indices of nonzero values in the array
    * format conversion, interaction with NumPy (`toarray()`)
    * ...
* attributes:
    * `mtx.T` - transpose (same as mtx.transpose())
    * `mtx.real` - real part of complex matrix
    * `mtx.imag` - imaginary part of complex matrix
    * `mtx.size` - the number of nonzeros (same as self.getnnz())
    * `mtx.shape` - the number of rows and columns (tuple)
* data and indices usually stored in 1D NumPy arrays

Sparse Array Classes
---------------------

.. toctree::
   :maxdepth: 2

   dia_array
   lil_array
   dok_array
   coo_array
   csr_array
   csc_array
   bsr_array

Summary
-------

.. list-table:: Summary of storage schemes.
   :widths: 10 10 10 10 10 10 10 30
   :header-rows: 1

   * - format
     - matrix * vector
     - get item
     - fancy get
     - set item
     - fancy set
     - solvers
     - note
   * - CSR
     - sparsetools
     - yes
     - yes
     - slow
     - .
     - any
     - has data array, fast row-wise ops
   * - CSC
     - sparsetools
     - yes
     - yes
     - slow
     - .
     - any
     - has data array, fast column-wise ops
   * - BSR
     - sparsetools
     - .
     - .
     - .
     - .
     - specialized
     - has data array, specialized
   * - COO
     - sparsetools
     - .
     - .
     - .
     - .
     - iterative
     - has data array, facilitates fast conversion
   * - DIA
     - sparsetools
     - .
     - .
     - .
     - .
     - iterative
     - has data array, specialized
   * - LIL
     - via CSR
     - yes
     - yes
     - yes
     - yes
     - iterative
     - arithmetic via CSR, incremental construction
   * - DOK
     - python
     - yes
     - one axis only
     - yes
     - yes
     - iterative
     - O(1) item access, incremental construction, slow arithmetic
