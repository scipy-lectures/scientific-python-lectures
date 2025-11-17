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

# List of Lists Format (LIL)

- row-based linked list
  - each row is a Python list (sorted) of column indices of non-zero elements
  - rows stored in a NumPy array (`dtype=np.object`)
  - non-zero values data stored analogously
- efficient for constructing sparse arrays incrementally
- constructor accepts:
  - dense array/matrix
  - sparse array/matrix
  - shape tuple (create empty array)
- flexible slicing, changing sparsity structure is efficient
- slow arithmetic, slow column slicing due to being row-based
- use:
  - when sparsity pattern is not known _apriori_ or changes
  - example: reading a sparse array from a text file

## Examples

### Create an empty LIL array:

```{code-cell}
mtx = sp.sparse.lil_array((4, 5))
```

### Prepare random data

```{code-cell}
rng = np.random.default_rng(27446968)
data = np.round(rng.random((2, 3)))
data
```

### Assign the data using fancy indexing

```{code-cell}
mtx[:2, [1, 2, 3]] = data
mtx
```

```{code-cell}
print(mtx)
```

```{code-cell}
mtx.toarray()
```

```{code-cell}
mtx.toarray()
```

### More slicing and indexing

```{code-cell}
mtx = sp.sparse.lil_array([[0, 1, 2, 0], [3, 0, 1, 0], [1, 0, 0, 1]])
mtx.toarray()
```

```{code-cell}
print(mtx)
```

```{code-cell}
mtx[:2, :]
```

```{code-cell}
mtx[:2, :].toarray()
```

```{code-cell}
mtx[1:2, [0,2]].toarray()
```

```{code-cell}
mtx.toarray()
```
