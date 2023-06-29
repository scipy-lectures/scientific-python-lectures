"""
Marker size
===========

Demo the marker size control in matplotlib.
"""

import matplotlib.pyplot as plt

size = 256, 16
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, 0, 1, 1], frameon=False)

for i in range(1, 11):
    plt.plot(
        [
            i,
        ],
        [
            1,
        ],
        "s",
        markersize=i,
        markerfacecolor="w",
        markeredgewidth=0.5,
        markeredgecolor="k",
    )

plt.xlim(0, 11)
plt.xticks([])
plt.yticks([])

plt.show()
