"""
Creating an image
==================

How to create an image with basic NumPy commands : ``np.zeros``, slicing...

This examples show how to create a simple checkerboard.
"""

import matplotlib.pyplot as plt
import numpy as np

check = np.zeros((8, 8))
check[::2, 1::2] = 1
check[1::2, ::2] = 1
plt.matshow(check, cmap="gray")
plt.show()
