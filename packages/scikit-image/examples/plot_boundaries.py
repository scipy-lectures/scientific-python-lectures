"""
Segmentation contours
=====================

Visualize segmentation contours on original grayscale image.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, segmentation

coins = data.coins()
mask = coins > filters.threshold_otsu(coins)
clean_border = segmentation.clear_border(mask).astype(int)

coins_edges = segmentation.mark_boundaries(coins, clean_border)

plt.figure(figsize=(8, 3.5))
plt.subplot(121)
plt.imshow(clean_border, cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.imshow(coins_edges)
plt.axis("off")

plt.tight_layout()
plt.show()
