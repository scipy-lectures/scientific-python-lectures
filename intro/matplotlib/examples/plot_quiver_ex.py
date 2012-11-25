import pylab as pl
import numpy as np

n = 8
X, Y = np.mgrid[0:n, 0:n]
T = np.arctan2(Y - n / 2., X - n/2.)
R = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
U, V = R * np.cos(T), R * np.sin(T)

pl.axes([0.025, 0.025, 0.95, 0.95])
pl.quiver(X, Y, U, V, R, alpha=.5)
pl.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)

pl.xlim(-1, n)
pl.xticks(())
pl.ylim(-1, n)
pl.yticks(())

pl.show()
