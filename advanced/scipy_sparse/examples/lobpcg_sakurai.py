"""
LOBPCG: block-preconditioned solver
===================================

This example demos the LOBPCG block-preconditioned solver.
"""

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, linewidth=90)


def sakurai(n):
    """Example taken from
    T. Sakurai, H. Tadano, Y. Inadomi and U. Nagashima
    A moment-based method for large-scale generalized eigenvalue problems
    Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004)
    """

    A = sp.sparse.eye(n, n)
    d0 = np.array(r_[5, 6 * ones(n - 2), 5])
    d1 = -4 * np.ones(n)
    d2 = np.ones(n)
    B = sp.sparse.spdiags([d2, d1, d0, d1, d2], [-2, -1, 0, 1, 2], n, n)

    k = np.arange(1, n + 1)
    w_ex = np.sort(
        1.0 / (16.0 * pow(np.cos(0.5 * k * np.pi / (n + 1)), 4))
    )  # exact eigenvalues

    return A, B, w_ex


m = 3  # Blocksize

#
# Large scale
#
n = 2500
A, B, w_ex = sakurai(n)  # Mikota pair
X = np.rand(n, m)
data = []
tt = time.clock()
eigs, vecs, resnh = sp.sparse.linalg.lobpcg(
    A, X, B, tol=1e-6, maxiter=500, retResidualNormsHistory=1
)
data.append(time.clock() - tt)
print("Results by LOBPCG for n=" + str(n))
print("")
print(eigs)
print("")
print("Exact eigenvalues")
print("")
print(w_ex[:m])
print("")
print("Elapsed time", data[0])
plt.loglog(np.arange(1, n + 1), w_ex, "b.")
plt.xlabel(r"Number $i$")
plt.ylabel(r"$\lambda_i$")
plt.title("Eigenvalue distribution")
plt.show()
