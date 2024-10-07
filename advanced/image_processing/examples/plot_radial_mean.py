"""
Radial mean
============

This example shows how to do a radial mean with scikit-image.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

f = sp.datasets.face(gray=True)
sx, sy = f.shape
X, Y = np.ogrid[0:sx, 0:sy]


r = np.hypot(X - sx / 2, Y - sy / 2)

rbin = (20 * r / r.max()).astype(int)
radial_mean = sp.ndimage.mean(f, labels=rbin, index=np.arange(1, rbin.max() + 1))

plt.figure(figsize=(5, 5))
plt.axes((0, 0, 1, 1))
plt.imshow(rbin, cmap="nipy_spectral")
plt.axis("off")

plt.show()
