"""
===================
Detrending a signal
===================

:func:`scipy.signal.detrend` removes a linear trend.
"""

############################################################
# Generate a random signal with a trend
import numpy as np

t = np.linspace(0, 5, 100)
rng = np.random.default_rng()
x = t + rng.normal(size=100)

############################################################
# Detrend
import scipy as sp

x_detrended = sp.signal.detrend(x)

############################################################
# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 4))
plt.plot(t, x, label="x")
plt.plot(t, x_detrended, label="x_detrended")
plt.legend(loc="best")
plt.show()
