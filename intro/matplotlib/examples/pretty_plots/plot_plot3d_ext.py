"""
3D plotting vignette
=====================

Demo 3D plotting with matplotlib and decorate the figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="hot")
ax.contourf(X, Y, Z, zdir="z", offset=-2, cmap="hot")

ax.set_zlim(-2, 2)
plt.xticks([])
plt.yticks([])
ax.set_zticks([])

ax.text2D(
    0.05,
    0.93,
    " 3D plots             \n",
    horizontalalignment="left",
    verticalalignment="top",
    size="xx-large",
    bbox={"facecolor": "white", "alpha": 1.0},
    transform=plt.gca().transAxes,
)

ax.text2D(
    0.05,
    0.87,
    " Plot 2D or 3D data",
    horizontalalignment="left",
    verticalalignment="top",
    size="large",
    transform=plt.gca().transAxes,
)

plt.show()
