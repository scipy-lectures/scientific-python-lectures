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

# Diagonal Format (DIA)

- very simple scheme
- diagonals in dense NumPy array of shape `(n_diag, length)`
  - fixed length -> waste space a bit when far from main diagonal
  - subclass of {class}`_data_matrix` (sparse array classes with
    `.data` attribute)
- offset for each diagonal
  - 0 is the main diagonal
  - negative offset = below
  - positive offset = above
- fast matrix \* vector (sparsetools)
- fast and easy item-wise operations
  - manipulate data array directly (fast NumPy machinery)
- constructor accepts:
  - dense array/matrix
  - sparse array/matrix
  - shape tuple (create empty array)
  - `(data, offsets)` tuple
- no slicing, no individual item access
- use:
  - rather specialized
  - solving PDEs by finite differences
  - with an iterative solver

## Examples

### Create some DIA arrays:

```{code-cell}
data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
data
```

```{code-cell}
offsets = np.array([0, -1, 2])
mtx = sp.sparse.dia_array((data, offsets), shape=(4, 4))
mtx
```

```{code-cell}
mtx.toarray()
```

```{code-cell}
data = np.arange(12).reshape((3, 4)) + 1
data
```

```{code-cell}
mtx = sp.sparse.dia_array((data, offsets), shape=(4, 4))
mtx.data
```

```{code-cell}
mtx.offsets
```

```{code-cell}
print(mtx)
```

```{code-cell}
mtx.toarray()
```

### Explanation with a scheme:

```
offset: row

    2:  9
    1:  --10------
    0:  1  . 11  .
    -1:  5  2  . 12
    -2:  .  6  3  .
    -3:  .  .  7  4
        ---------8
```

### Matrix-vector multiplication

```{code-cell}
vec = np.ones((4, ))
vec
```

```{code-cell}
mtx @ vec
```

```{code-cell}
(mtx * vec).toarray()
```
