"""
Equalizing the histogram of an image
=====================================

Histogram equalizing makes images have a uniform histogram.
"""

import matplotlib.pyplot as plt
from skimage import data, exposure

camera = data.camera()
camera_equalized = exposure.equalize_hist(camera)

plt.figure(figsize=(7, 3))

plt.subplot(121)
plt.imshow(camera, cmap="gray", interpolation="nearest")
plt.axis("off")
plt.subplot(122)
plt.imshow(camera_equalized, cmap="gray", interpolation="nearest")
plt.axis("off")
plt.tight_layout()
plt.show()
