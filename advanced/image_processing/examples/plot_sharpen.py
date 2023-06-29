"""
Image sharpening
=================

This example shows how to sharpen an image in noiseless situation by
applying the filter inverse to the blur.
"""

import scipy as sp
import matplotlib.pyplot as plt

f = sp.datasets.face(gray=True).astype(float)
blurred_f = sp.ndimage.gaussian_filter(f, 3)

filter_blurred_f = sp.ndimage.gaussian_filter(blurred_f, 1)

alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(f, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(132)
plt.imshow(blurred_f, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(133)
plt.imshow(sharpened, cmap=plt.cm.gray)
plt.axis("off")

plt.tight_layout()
plt.show()
