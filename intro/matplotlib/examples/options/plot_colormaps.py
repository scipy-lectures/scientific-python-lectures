"""
Colormaps
=========

An example plotting the matplotlib colormaps.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc("text", usetex=False)
a = np.outer(np.arange(0, 1, 0.01), np.ones(10))

plt.figure(figsize=(10, 5))
plt.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.99)
maps = [m for m in plt.cm.datad if not m.endswith("_r")]
maps.sort()
l = len(maps) + 1

for i, m in enumerate(maps):
    plt.subplot(1, l, i + 1)
    plt.axis("off")
    plt.imshow(a, aspect="auto", cmap=plt.get_cmap(m), origin="lower")
    plt.title(m, rotation=90, fontsize=10, va="bottom")

plt.show()
