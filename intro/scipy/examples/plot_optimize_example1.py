"""
=========================================
Finding the minimum of a smooth function
=========================================

Demos various methods to find the minimum of a function.
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2 + 10 * np.sin(x)


x = np.arange(-5, 5, 0.1)
plt.plot(x, f(x))

############################################################
# Now find the minimum with a few methods
import scipy as sp

# The default (Nelder Mead)
print(sp.optimize.minimize(f, x0=0))

############################################################

plt.show()
