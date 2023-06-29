"""
===============================
Minima and roots of a function
===============================

Demos finding minima and roots of a function.
"""

############################################################
# Define the function
############################################################

import numpy as np

x = np.arange(-10, 10, 0.1)


def f(x):
    return x**2 + 10 * np.sin(x)


############################################################
# Find minima
############################################################

import scipy as sp

# Global optimization
grid = (-10, 10, 0.1)
xmin_global = sp.optimize.brute(f, (grid,))
print(f"Global minima found {xmin_global}")

# Constrain optimization
xmin_local = sp.optimize.fminbound(f, 0, 10)
print(f"Local minimum found {xmin_local}")

############################################################
# Root finding
############################################################

root = sp.optimize.root(f, 1)  # our initial guess is 1
print(f"First root found {root.x}")
root2 = sp.optimize.root(f, -2.5)
print(f"Second root found {root2.x}")

############################################################
# Plot function, minima, and roots
############################################################

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

# Plot the function
ax.plot(x, f(x), "b-", label="f(x)")

# Plot the minima
xmins = np.array([xmin_global[0], xmin_local])
ax.plot(xmins, f(xmins), "go", label="Minima")

# Plot the roots
roots = np.array([root.x, root2.x])
ax.plot(roots, f(roots), "kv", label="Roots")

# Decorate the figure
ax.legend(loc="best")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.axhline(0, color="gray")
plt.show()
