import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = pl.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=pl.cm.hot)
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=pl.cm.hot)
ax.set_zlim(-2, 2)
pl.xticks(())
pl.yticks(())
ax.set_zticks(())

ax.text2D(0.05, .93, " 3D plots \n",
          horizontalalignment='left',
          verticalalignment='top',
          size='xx-large',
          bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
          transform=pl.gca().transAxes)

ax.text2D(0.05, .87, " Plot 2D or 3D data",
          horizontalalignment='left',
          verticalalignment='top',
          size='large',
          transform=pl.gca().transAxes)

pl.show()
