"""
Solve a linear system
=======================

Construct a 1000x1000 lil_array and add some values to it, convert it
to CSR format and solve A x = b for x:and solve a linear system with a
direct solver.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

rng = np.random.default_rng(27446968)

mtx = sp.sparse.lil_array((1000, 1000), dtype=np.float64)
mtx[0, :100] = rng.random(100)
mtx[1, 100:200] = mtx[0, :100]
mtx.setdiag(rng.random(1000))

plt.clf()
plt.spy(mtx, marker=".", markersize=2)
plt.show()

mtx = mtx.tocsr()
rhs = rng.random(1000)

x = sp.sparse.linalg.spsolve(mtx, rhs)

print(f"residual: {np.linalg.norm(mtx * x - rhs)!r}")
