---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.0-dev
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
orphan: true
---

```{code-cell}
:tags: [hide-input]

import numpy as np
import scipy as sp
```

# Block Compressed Row Format (BSR)

- basically a CSR with dense sub-matrices of fixed shape instead of scalar
  items

  - block size `(R, C)` must evenly divide the shape of the matrix `(M, N)`
  - three NumPy arrays: `indices`, `indptr`, `data`

    - `indices` is array of column indices for each block

    - `data` is array of corresponding nonzero values of shape `(nnz, R, C)`

    - ...

  - subclass of {class}`_cs_matrix` (common CSR/CSC functionality)
    - subclass of {class}`_data_matrix` (sparse matrix classes with
      `.data` attribute)

- fast matrix vector products and other arithmetic (sparsetools)
- constructor accepts:
  - dense array/matrix
  - sparse array/matrix
  - shape tuple (create empty array)
  - `(data, coords)` tuple
  - `(data, indices, indptr)` tuple
- many arithmetic operations considerably more efficient than CSR for
  sparse matrices with dense sub-matrices
- use:
  - like CSR
  - vector-valued finite element discretizations

## Examples

### Create empty BSR array with (1, 1) block size (like CSR...):

```{code-cell}
mtx = sp.sparse.bsr_array((3, 4), dtype=np.int8)
mtx
```

```{code-cell}
mtx.toarray()
```

### Create empty BSR array with (3, 2) block size:

```{code-cell}
mtx = sp.sparse.bsr_array((3, 4), blocksize=(3, 2), dtype=np.int8)
mtx
```

```{code-cell}
mtx.toarray()
```

<!---
# What does this refer to?

- a bug?

-->

### Create using `(data, coords)` tuple with (1, 1) block size (like CSR...):

```{code-cell}
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
mtx = sp.sparse.bsr_array((data, (row, col)), shape=(3, 3))
mtx
```

```{code-cell}
mtx.toarray()
```

```{code-cell}
mtx.data
```

```{code-cell}
mtx.indices
```

```{code-cell}
mtx.indptr
```

### Create using `(data, indices, indptr)` tuple with (2, 2) block size:

```{code-cell}
indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
mtx = sp.sparse.bsr_array((data, indices, indptr), shape=(6, 6))
mtx.toarray()
```

```{code-cell}
data
```
