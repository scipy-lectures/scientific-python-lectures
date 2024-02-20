"""
Opening, erosion, and propagation
==================================

This example shows simple operations of mathematical morphology.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

square = np.zeros((32, 32))
square[10:-10, 10:-10] = 1
rng = np.random.default_rng(27446968)
x, y = (32 * rng.random((2, 20))).astype(int)
square[x, y] = 1

open_square = sp.ndimage.binary_opening(square)

eroded_square = sp.ndimage.binary_erosion(square)
reconstruction = sp.ndimage.binary_propagation(eroded_square, mask=square)

plt.figure(figsize=(9.5, 3))
plt.subplot(131)
plt.imshow(square, cmap=plt.cm.gray, interpolation="nearest")
plt.axis("off")
plt.subplot(132)
plt.imshow(open_square, cmap=plt.cm.gray, interpolation="nearest")
plt.axis("off")
plt.subplot(133)
plt.imshow(reconstruction, cmap=plt.cm.gray, interpolation="nearest")
plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0.02, top=0.99, bottom=0.01, left=0.01, right=0.99)
plt.show()
