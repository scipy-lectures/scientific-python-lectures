"""
A simple example
=================

"""

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 100)
Y = np.sin(X)

plt.plot(X, Y, linewidth=2)
plt.show()
