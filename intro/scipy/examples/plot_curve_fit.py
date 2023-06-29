"""
===============
Curve fitting
===============

Demos a simple curve fitting
"""

############################################################
# First generate some data
import numpy as np

# Seed the random number generator for reproducibility
rng = np.random.default_rng(27446968)

x_data = np.linspace(-5, 5, num=50)
noise = 0.01 * np.cos(100 * x_data)
a, b = 2.9, 1.5
y_data = a * np.cos(b * x_data) + noise

# And plot it
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data)

############################################################
# Now fit a simple sine function to the data
import scipy as sp


def test_func(x, a, b, c):
    return a * np.sin(b * x + c)


params, params_covariance = sp.optimize.curve_fit(
    test_func, x_data, y_data, p0=[2, 1, 3]
)

print(params)

############################################################
# And plot the resulting curve on the data

plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, label="Data")
plt.plot(x_data, test_func(x_data, *params), label="Fitted function")

plt.legend(loc="best")

plt.show()
