import pylab as pl
import numpy as np

n = 16
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
pl.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
pl.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
pl.xlim(-.5, n)
pl.xticks(())
pl.ylim(-1, 1)
pl.yticks(())

pl.text(-0.05, 1.02, " Bar Plot:              pl.bar(...)\n",
      horizontalalignment='left',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
      transform=pl.gca().transAxes)

pl.text(-0.05, 1.01, "\n\n   Make a bar plot with rectangles ",
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      transform=pl.gca().transAxes)

pl.show()
