"""
Blurring of images
===================

An example showing various processes that blur an image.
"""

import scipy as sp
import matplotlib.pyplot as plt

face = sp.datasets.face(gray=True)
blurred_face = sp.ndimage.gaussian_filter(face, sigma=3)
very_blurred = sp.ndimage.gaussian_filter(face, sigma=5)
local_mean = sp.ndimage.uniform_filter(face, size=11)

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.imshow(blurred_face, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(132)
plt.imshow(very_blurred, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(133)
plt.imshow(local_mean, cmap=plt.cm.gray)
plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0.0, top=0.99, bottom=0.01, left=0.01, right=0.99)

plt.show()
