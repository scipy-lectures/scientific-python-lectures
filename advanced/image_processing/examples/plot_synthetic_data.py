"""
Synthetic data
===============

The example generates and displays simple synthetic data.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

rng = np.random.default_rng(27446968)
n = 10
l = 256
im = np.zeros((l, l))
points = l * rng.random((2, n**2))
im[(points[0]).astype(int), (points[1]).astype(int)] = 1
im = sp.ndimage.gaussian_filter(im, sigma=l / (4.0 * n))

mask = im > im.mean()

label_im, nb_labels = sp.ndimage.label(mask)

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.imshow(im)
plt.axis("off")
plt.subplot(132)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(133)
plt.imshow(label_im, cmap=plt.cm.nipy_spectral)
plt.axis("off")

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
plt.show()
