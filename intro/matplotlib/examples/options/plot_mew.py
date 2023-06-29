"""
Marker edge width
=================

Demo the marker edge widths of matplotlib's markers.
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
        markersize=5,
        markeredgewidth=1 + i / 10.0,
        markeredgecolor="k",
        markerfacecolor="w",
    )
plt.xlim(0, 11)
plt.xticks([])
plt.yticks([])

plt.show()
