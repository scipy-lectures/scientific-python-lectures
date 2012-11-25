import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

fig = pl.figure(figsize=(5, 4), dpi=72)
axes = fig.add_axes([0.01, 0.01, .98, 0.98])
X = np.linspace(0, 2, 200, endpoint=True)
Y = np.sin(2 * np.pi * X)
pl.plot(X, Y, lw=.25, c='k')
pl.xticks(np.arange(0.0, 2.0, 0.1))
pl.yticks(np.arange(-1.0, 1.0, 0.1))
pl.grid()
pl.show()
