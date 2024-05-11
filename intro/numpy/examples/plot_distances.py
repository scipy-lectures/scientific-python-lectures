"""
Distances exercise
==================

Plot distances in a grid

"""

import matplotlib.pyplot as plt
import numpy as np

x, y = np.arange(5), np.arange(5)[:, np.newaxis]
distance = np.sqrt(x**2 + y**2)
plt.pcolor(distance)
plt.colorbar()
plt.show()
