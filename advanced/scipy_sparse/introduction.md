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
---

# Scipy sparse arrays

**Section author**: _Robert Cimrman_

(Dense) matrix is:

- mathematical object
- data structure for storing a 2D array of values

Important features:

- memory allocated once for all items
  - usually a contiguous chunk, think NumPy ndarray
- _fast_ access to individual items (\*)

## Why Sparse Matrices?

- the memory grows like `n**2` for dense matrix

- small example (double precision matrix):

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1e6, 10)
plt.plot(x, 8.0 * (x**2) / 1e6, lw=5)
plt.xlabel('size n')
plt.ylabel('memory [MB]')
```

## Sparse Matrices vs. Sparse Matrix Storage Schemes

- sparse matrix is a matrix, which is _almost empty_
- storing all the zeros is wasteful -> store only nonzero items
- think **compression**
- pros: huge memory savings
- cons: slow access to individual items, but it depends on actual storage scheme.

## Typical Applications

- solution of partial differential equations (PDEs)

  - the _finite element method_
  - mechanical engineering, electrotechnics, physics, ...

- graph theory

  - nonzero at `(i, j)` means that node `i` is connected to node `j`

- natural language processing

  - nonzero at `(i, j)` means that the document `i` contains the word `j`

- ...

:::{admonition} Prerequisites

- {ref}`numpy <numpy>`
- {ref}`scipy <scipy>`
- {ref}`matplotlib (optional) <matplotlib>`
- {ref}`ipython (the enhancements come handy) <interactive-work>`
  :::

## Sparsity Structure Visualization

- {func}`spy` from `matplotlib`
- example plots:

![](figures/graph.png)

![](figures/graph_g.png)

![](figures/graph_rcm.png)
