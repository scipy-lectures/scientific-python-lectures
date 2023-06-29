"""
=============================
Demo mathematical morphology
=============================

A basic demo of binary opening and closing.
"""

# Generate some binary data
import numpy as np

np.random.seed(0)
a = np.zeros((50, 50))
a[10:-10, 10:-10] = 1
a += 0.25 * np.random.standard_normal(a.shape)
mask = a >= 0.5

# Apply mathematical morphology
import scipy as sp

opened_mask = sp.ndimage.binary_opening(mask)
closed_mask = sp.ndimage.binary_closing(opened_mask)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 3.5))
plt.subplot(141)
plt.imshow(a, cmap=plt.cm.gray)
plt.axis("off")
plt.title("a")

plt.subplot(142)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis("off")
plt.title("mask")

plt.subplot(143)
plt.imshow(opened_mask, cmap=plt.cm.gray)
plt.axis("off")
plt.title("opened_mask")

plt.subplot(144)
plt.imshow(closed_mask, cmap=plt.cm.gray)
plt.title("closed_mask")
plt.axis("off")

plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.99)

plt.show()
