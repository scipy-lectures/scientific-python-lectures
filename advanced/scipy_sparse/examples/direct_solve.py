"""
Solve a linear system
=======================

Construct a 1000x1000 lil_matrix and add some values to it, convert it
to CSR format and solve A x = b for x:and solve a linear system with a
direct solver.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

rand = np.random.rand

mtx = sp.sparse.lil_matrix((1000, 1000), dtype=np.float64)
mtx[0, :100] = rand(100)
mtx[1, 100:200] = mtx[0, :100]
mtx.setdiag(rand(1000))

plt.clf()
plt.spy(mtx, marker='.', markersize=2)
plt.show()

mtx = mtx.tocsr()
rhs = rand(1000)

x = sp.sparse.linalg.spsolve(mtx, rhs)

print(f'rezidual: {np.linalg.norm(mtx * x - rhs)!r}')
