"""
Solid joint style
==================

An example showing the different solid joint styles in matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

size = 256, 16
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, 0, 1, 1], frameon=False)

plt.plot(np.arange(3), [0, 1, 0], color="blue", linewidth=8, solid_joinstyle="miter")
plt.plot(
    4 + np.arange(3), [0, 1, 0], color="blue", linewidth=8, solid_joinstyle="bevel"
)
plt.plot(
    8 + np.arange(3), [0, 1, 0], color="blue", linewidth=8, solid_joinstyle="round"
)

plt.xlim(0, 12)
plt.ylim(-1, 2)
plt.xticks([])
plt.yticks([])

plt.show()
