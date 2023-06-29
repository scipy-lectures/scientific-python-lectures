"""
Dash capstyle
=============

An example demoing the dash capstyle.
"""

import numpy as np
import matplotlib.pyplot as plt

size = 256, 16
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, 0, 1, 1], frameon=False)

plt.plot(
    np.arange(4),
    np.ones(4),
    color="blue",
    dashes=[15, 15],
    linewidth=8,
    dash_capstyle="butt",
)

plt.plot(
    5 + np.arange(4),
    np.ones(4),
    color="blue",
    dashes=[15, 15],
    linewidth=8,
    dash_capstyle="round",
)

plt.plot(
    10 + np.arange(4),
    np.ones(4),
    color="blue",
    dashes=[15, 15],
    linewidth=8,
    dash_capstyle="projecting",
)

plt.xlim(0, 14)
plt.xticks([])
plt.yticks([])

plt.show()
