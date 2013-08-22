"""
Visiualize segmentation contours on original grayscale image.
"""

from skimage import data, segmentation, filter, color
import matplotlib.pyplot as plt

coins = data.coins()
mask = coins > filter.threshold_otsu(coins)
clean_border = segmentation.clear_border(mask)

coins_edges = segmentation.visualize_boundaries(color.gray2rgb(coins),
                            clean_border)

plt.figure(figsize=(8, 3.5))
plt.subplot(121)
plt.imshow(clean_border, cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(coins_edges)
plt.axis('off')

plt.tight_layout()
plt.show()
