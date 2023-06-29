"""
Plotting quiver decorated
==========================

An example showing quiver with decorations.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 8
X, Y = np.mgrid[0:n, 0:n]
T = np.arctan2(Y - n / 2.0, X - n / 2.0)
R = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
U, V = R * np.cos(T), R * np.sin(T)

plt.quiver(X, Y, U, V, R, alpha=0.5)
plt.quiver(X, Y, U, V, edgecolor="k", facecolor="None", linewidth=0.5)

plt.xlim(-1, n)
plt.xticks([])
plt.ylim(-1, n)
plt.yticks([])


# Add a title and a box around it
from matplotlib.patches import FancyBboxPatch

ax = plt.gca()
ax.add_patch(
    FancyBboxPatch(
        (-0.05, 0.87),
        width=0.66,
        height=0.165,
        clip_on=False,
        boxstyle="square,pad=0",
        zorder=3,
        facecolor="white",
        alpha=1.0,
        transform=plt.gca().transAxes,
    )
)

plt.text(
    -0.05,
    1.02,
    " Quiver Plot:    plt.quiver(...)\n",
    horizontalalignment="left",
    verticalalignment="top",
    size="xx-large",
    transform=plt.gca().transAxes,
)

plt.text(
    -0.05,
    1.01,
    "\n\n    Plot a 2-D field of arrows ",
    horizontalalignment="left",
    verticalalignment="top",
    size="large",
    transform=plt.gca().transAxes,
)


plt.show()
