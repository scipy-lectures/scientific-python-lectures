"""
This example shows how to label connected components of a binary image, using
the dedicated skimage.morphology.label function.
"""

from skimage import morphology
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

n = 12
l = 256
np.random.seed(1)
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
blobs = im > 0.7 * im.mean()

all_labels = morphology.label(blobs)
blobs_labels = morphology.label(blobs, background=0)

plt.figure(figsize=(9, 3.5))
plt.subplot(131)
plt.imshow(blobs, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(all_labels)
plt.axis('off')
plt.subplot(133)
plt.imshow(blobs_labels)
plt.axis('off')

plt.tight_layout()
plt.show()
