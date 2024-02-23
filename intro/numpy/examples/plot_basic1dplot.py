"""
1D plotting
===========

Plot a basic 1D figure

"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 20)
y = np.linspace(0, 9, 20)
plt.plot(x, y)
plt.plot(x, y, "o")
plt.show()
