import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

fig = pl.figure(figsize=(8, 5), dpi=72)
fig.patch.set_alpha(0.0)
axes = pl.subplot(111)

n = 5
Z = np.zeros((n, 4))
X = np.linspace(0, 2, n, endpoint=True)
Y = np.random.random((n, 4))
pl.boxplot(Y)

pl.xticks(()), pl.yticks(())

pl.text(-0.05, 1.02, " Box Plot:   pl.boxplot(...)\n",
        horizontalalignment='left',
        verticalalignment='top',
        size='xx-large',
        bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
        transform=axes.transAxes)

pl.text(-0.05, 1.01, " Make a box and whisker plot ",
        horizontalalignment='left',
        verticalalignment='top',
        size='large',
        transform=axes.transAxes)

pl.show()
