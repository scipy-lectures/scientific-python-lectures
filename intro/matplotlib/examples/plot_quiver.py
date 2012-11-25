import pylab as pl
import numpy as np

n = 8
X, Y = np.mgrid[0:n, 0:n]
T = np.arctan2(Y - n/ 2., X - n / 2.)
R = 10 + np.sqrt((Y - n / 2.) ** 2 + (X - n / 2.) ** 2)
U, V = R * np.cos(T), R * np.sin(T)

pl.quiver(X, Y, U, V, R, alpha=.5)
pl.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)

pl.xlim(-1, n)
pl.xticks(())
pl.ylim(-1, n)
pl.yticks(())

pl.text(-0.05, 1.02, " Quiver Plot:    pl.quiver(...)\n",
      horizontalalignment='left',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
      transform=pl.gca().transAxes)

pl.text(-0.05, 1.01, "\n\n    Plot a 2-D field of arrows ",
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      transform=pl.gca().transAxes)

pl.show()
