"""
Display a Raccoon Face
======================

An example that displays a raccoon face with matplotlib.
"""

import scipy as sp
import matplotlib.pyplot as plt

f = sp.datasets.face(gray=True)

plt.figure(figsize=(10, 3.6))

plt.subplot(131)
plt.imshow(f, cmap="gray")

plt.subplot(132)
plt.imshow(f, cmap="gray", vmin=30, vmax=200)
plt.axis("off")

plt.subplot(133)
plt.imshow(f, cmap="gray")
plt.contour(f, [50, 200])
plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0.0, top=0.99, bottom=0.01, left=0.05, right=0.99)
plt.show()
