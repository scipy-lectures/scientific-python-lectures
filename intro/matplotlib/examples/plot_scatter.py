import pylab as pl
import numpy as np

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)

T = np.arctan2(Y,X)

pl.scatter(X, Y, s=75, c=T, alpha=.5)
pl.xlim(-1.5, 1.5)
pl.xticks(())
pl.ylim(-1.5, 1.5)
pl.yticks(())

pl.text(-0.05, 1.02, " Scatter Plot:  pl.scatter(...)\n",
      horizontalalignment='left',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
      transform=pl.gca().transAxes)

pl.text(-0.05, 1.01, "\n\n   Make a scatter plot of x versus y ",
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      transform=pl.gca().transAxes)

pl.show()
