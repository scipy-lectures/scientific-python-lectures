"""
3D plotting
===========

A simple example of 3D plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax: Axes3D = plt.figure().add_subplot(projection="3d")
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="hot")
ax.contourf(X, Y, Z, zdir="z", offset=-2, cmap="hot")
ax.set_zlim(-2, 2)

plt.show()
