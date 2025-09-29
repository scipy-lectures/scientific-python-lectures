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

# Dictionary of Keys Format (DOK)

- subclass of Python dict
  - keys are `(row, column)` index tuples (no duplicate entries allowed)
  - values are corresponding non-zero values
- efficient for constructing sparse arrays incrementally
- constructor accepts:
  - dense array/matrix
  - sparse array/matrix
  - shape tuple (create empty array)
- efficient O(1) access to individual elements
- flexible slicing, changing sparsity structure is efficient
- can be efficiently converted to a coo_array once constructed
- slow arithmetic (`for` loops with `dict.items()`)
- use:
  - when sparsity pattern is not known apriori or changes

## Examples

### Create a DOK array element by element:

```{code-cell}
mtx = sp.sparse.dok_array((5, 5), dtype=np.float64)
mtx
```

```{code-cell}
for ir in range(5):
    for ic in range(5):
        mtx[ir, ic] = 1.0 * (ir != ic)
mtx
```

```{code-cell}
mtx.toarray()
```

### Slicing and indexing:

```{code-cell}
mtx[1, 1]
```

```{code-cell}
mtx[[1], 1:3]
```

```{code-cell}
mtx[[1], 1:3].toarray()
```

```{code-cell}
mtx[[2, 1], 1:3].toarray()
```
