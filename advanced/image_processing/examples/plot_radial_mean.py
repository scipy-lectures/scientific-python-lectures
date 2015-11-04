"""
Radial mean
============

This example shows how to do a radial mean with scikit-image.
"""

import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

f = scipy.misc.face(gray=True)
sx, sy = f.shape
X, Y = np.ogrid[0:sx, 0:sy]


r = np.hypot(X - sx/2, Y - sy/2)

rbin = (20* r/r.max()).astype(np.int)
radial_mean = ndimage.mean(f, labels=rbin, index=np.arange(1, rbin.max() +1))

plt.figure(figsize=(5, 5))
plt.axes([0, 0, 1, 1])
plt.imshow(rbin, cmap=plt.cm.spectral)
plt.axis('off')

plt.show()
