"""
============================
A demo of 1D interpolation
============================

"""

# Generate data
import numpy as np

rng = np.random.default_rng(27446968)
measured_time = np.linspace(0, 1, 10)
noise = 1e-1 * (rng.random(10) * 2 - 1)
measures = np.sin(2 * np.pi * measured_time) + noise

# Interpolate it to new time points
import scipy as sp

linear_interp = sp.interpolate.interp1d(measured_time, measures)
interpolation_time = np.linspace(0, 1, 50)
linear_results = linear_interp(interpolation_time)
cubic_interp = sp.interpolate.interp1d(measured_time, measures, kind="cubic")
cubic_results = cubic_interp(interpolation_time)

# Plot the data and the interpolation
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(measured_time, measures, "o", ms=6, label="measures")
plt.plot(interpolation_time, linear_results, label="linear interp")
plt.plot(interpolation_time, cubic_results, label="cubic interp")
plt.legend()
plt.show()
