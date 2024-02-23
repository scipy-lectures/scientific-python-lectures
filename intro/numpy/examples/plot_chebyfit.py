"""
Fitting in Chebyshev basis
==========================

Plot noisy data and their polynomial fit in a Chebyshev basis

"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(27446968)

x = np.linspace(-1, 1, 2000)
y = np.cos(x) + 0.3 * rng.random(2000)
p = np.polynomial.Chebyshev.fit(x, y, 90)

plt.plot(x, y, "r.")
plt.plot(x, p(x), "k-", lw=3)
plt.show()
