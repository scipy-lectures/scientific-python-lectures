"""
3D plotting
============

Demo 3D plotting with matplotlib and style the figure.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

ax = plt.figure().add_subplot(projection="3d")
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contourf(X, Y, Z)
ax.clabel(cset, fontsize=9, inline=1)

plt.xticks([])
plt.yticks([])
ax.set_zticks([])


ax.text2D(
    -0.05,
    1.05,
    " 3D plots             \n",
    horizontalalignment="left",
    verticalalignment="top",
    bbox={"facecolor": "white", "alpha": 1.0},
    family="DejaVu Sans",
    size="x-large",
    transform=plt.gca().transAxes,
)

ax.text2D(
    -0.05,
    0.975,
    " Plot 2D or 3D data",
    horizontalalignment="left",
    verticalalignment="top",
    family="DejaVu Sans",
    size="medium",
    transform=plt.gca().transAxes,
)

plt.show()
