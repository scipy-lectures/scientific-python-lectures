Storage Schemes
===============

* seven sparse matrix types in scipy.sparse:
    1. csc_matrix: Compressed Sparse Column format
    2. csr_matrix: Compressed Sparse Row format
    3. bsr_matrix: Block Sparse Row format
    4. lil_matrix: List of Lists format
    5. dok_matrix: Dictionary of Keys format
    6. coo_matrix: COOrdinate format (aka IJV, triplet format)
    7. dia_matrix: DIAgonal format
* each suitable for some tasks
* many employ sparsetools C++ module by Nathan Bell
* assume the following is imported::
    
  >>> import numpy as np
  >>> import scipy.sparse as sps
  >>> import matplotlib.pyplot as plt

* **warning** for NumPy users:
    * the multiplication with '*' is the *matrix multiplication* (dot product)
    * not part of NumPy!
        * passing a sparse matrix object to NumPy functions expecting
          ndarray/matrix does not work

Common Methods
--------------

* all scipy.sparse classes are subclasses of :class:`spmatrix`
    * default implementation of arithmetic operations
        * always converts to CSR
        * subclasses override for efficiency
    * shape, data type set/get
    * nonzero indices
    * format conversion, interaction with NumPy (`toarray()`, `todense()`)
    * ...
* attributes:
    * `mtx.A` - same as mtx.toarray()
    * `mtx.T` - transpose (same as mtx.transpose())
    * `mtx.H` - Hermitian (conjugate) transpose
    * `mtx.real` - real part of complex matrix
    * `mtx.imag` - imaginary part of complex matrix
    * `mtx.size` - the number of nonzeros (same as self.getnnz())
    * `mtx.shape` - the number of rows and columns (tuple)
* data usually stored in NumPy arrays

Sparse Matrix Classes
---------------------

.. toctree::
   :maxdepth: 2

   dia_matrix
   lil_matrix
   dok_matrix
   coo_matrix
   csr_matrix
   csc_matrix
   bsr_matrix

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
     - arithmetics via CSR, incremental construction
   * - DOK
     - python
     - yes
     - one axis only
     - yes
     - yes
     - iterative
     - O(1) item access, incremental construction
   * - COO
     - sparsetools
     - .
     - .
     - .
     - .
     - iterative
     - has data array, facilitates fast conversion
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
