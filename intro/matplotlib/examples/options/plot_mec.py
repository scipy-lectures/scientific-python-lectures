"""
Marker edge color
==================

Demo the marker edge color of matplotlib's markers.
"""

import numpy as np
import matplotlib.pyplot as plt

size = 256, 16
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, 0, 1, 1], frameon=False)

rng = np.random.default_rng()

for i in range(1, 11):
    r, g, b = np.random.uniform(0, 1, 3)
    plt.plot(
        [
            i,
        ],
        [
            1,
        ],
        "s",
        markersize=5,
        markerfacecolor="w",
        markeredgewidth=1.5,
        markeredgecolor=(r, g, b, 1),
    )

plt.xlim(0, 11)
plt.xticks([])
plt.yticks([])

plt.show()
