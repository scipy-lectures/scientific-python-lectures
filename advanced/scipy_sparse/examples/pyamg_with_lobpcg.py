"""
Compute eigenvectors and eigenvalues using a preconditioned eigensolver
=======================================================================

In this example Smoothed Aggregation (SA) is used to precondition
the LOBPCG eigensolver on a two-dimensional Poisson problem with
Dirichlet boundary conditions.
"""

import scipy as sp
import matplotlib.pyplot as plt

from pyamg import smoothed_aggregation_solver
from pyamg.gallery import poisson

N = 100
K = 9
A = poisson((N, N), format="csr")

# create the AMG hierarchy
ml = smoothed_aggregation_solver(A)

# initial approximation to the K eigenvectors
X = sp.rand(A.shape[0], K)

# preconditioner based on ml
M = ml.aspreconditioner()

# compute eigenvalues and eigenvectors with LOBPCG
W, V = sp.sparse.linalg.lobpcg(A, X, M=M, tol=1e-8, largest=False)


# plot the eigenvectors
plt.figure(figsize=(9, 9))

for i in range(K):
    plt.subplot(3, 3, i + 1)
    plt.title("Eigenvector %d" % i)
    plt.pcolor(V[:, i].reshape(N, N))
    plt.axis("equal")
    plt.axis("off")
plt.show()
