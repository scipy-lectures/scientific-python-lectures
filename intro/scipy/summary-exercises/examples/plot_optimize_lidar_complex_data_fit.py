"""
The lidar system, data and fit (2 of 2 datasets)
================================================

Generate a chart of the data fitted by Gaussian curve
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def model(t, coeffs):
    return (
        coeffs[0]
        + coeffs[1] * np.exp(-(((t - coeffs[2]) / coeffs[3]) ** 2))
        + coeffs[4] * np.exp(-(((t - coeffs[5]) / coeffs[6]) ** 2))
        + coeffs[7] * np.exp(-(((t - coeffs[8]) / coeffs[9]) ** 2))
    )


def residuals(coeffs, y, t):
    return y - model(t, coeffs)


waveform_2 = np.load("waveform_2.npy")
t = np.arange(len(waveform_2))

x0 = np.array([3, 30, 20, 1, 12, 25, 1, 8, 28, 1], dtype=float)
x, flag = sp.optimize.leastsq(residuals, x0, args=(waveform_2, t))

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(t, waveform_2, t, model(t, x))
plt.xlabel("Time [ns]")
plt.ylabel("Amplitude [bins]")
plt.legend(["Waveform", "Model"])
plt.show()
