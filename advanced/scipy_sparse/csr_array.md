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

# Compressed Sparse Row Format (CSR)

- row oriented
  - three NumPy arrays: `indices`, `indptr`, `data`
    - `indices` is array of column indices
    - `data` is array of corresponding nonzero values
    - `indptr` points to row starts in `indices` and `data`
    - length of `indptr` is `n_row + 1`,
      last item = number of values = length of both `indices` and `data`
    - nonzero values of the `i`-th row are `data[indptr[i]:indptr[i + 1]]`
      with column indices `indices[indptr[i]:indptr[i + 1]]`
    - item `(i, j)` can be accessed as `data[indptr[i] + k]`, where `k` is
      position of `j` in `indices[indptr[i]:indptr[i + 1]]`
  - subclass of {class}`_cs_matrix` (common CSR/CSC functionality)
    - subclass of {class}`_data_matrix` (sparse array classes with
      `.data` attribute)
- fast matrix vector products and other arithmetic (sparsetools)
- constructor accepts:
  - dense array/matrix
  - sparse array/matrix
  - shape tuple (create empty array)
  - `(data, coords)` tuple
  - `(data, indices, indptr)` tuple
- efficient row slicing, row-oriented operations
- slow column slicing, expensive changes to the sparsity structure
- use:
  - actual computations (most linear solvers support this format)

## Examples

### Create empty CSR array:

```{code-cell}
mtx = sp.sparse.csr_array((3, 4), dtype=np.int8)
mtx.toarray()
```

### Create using `(data, coords)` tuple:

```{code-cell}
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
mtx = sp.sparse.csr_array((data, (row, col)), shape=(3, 3))
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

### Create using `(data, indices, indptr)` tuple:

```{code-cell}
data = np.array([1, 2, 3, 4, 5, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
indptr = np.array([0, 2, 3, 6])
mtx = sp.sparse.csr_array((data, indices, indptr), shape=(3, 3))
mtx.toarray()
```
