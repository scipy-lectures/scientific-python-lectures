import pylab as pl
import numpy as np

def marker(m, i):
    X = i * .5 * np.ones(11)
    Y = np.arange(11)

    pl.plot(X, Y, color='None', lw=1, marker=m, ms=10, mfc=(.75, .75, 1, 1),
            mec=(0, 0, 1, 1))
    pl.text(.5 * i, 10.25, repr(m), rotation=90, fontsize=15, va='bottom')

markers = [0, 1, 2, 3, 4, 5, 6, 7, 'o', 'h', '_', '1', '2', '3', '4',
          '8', 'p', '^', 'v', '<', '>', '|', 'd', ',', '+', 's', '*',
          '|', 'x', 'D', 'H', '.']

n_markers = len(markers)

size = 20 * n_markers, 300
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = pl.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
pl.axes([0, 0.01, 1, .9], frameon=False)

for i, m in enumerate(markers):
    marker(m, i)

pl.xlim(-.2, .2 + .5 * n_markers)
pl.xticks(())
pl.yticks(())

pl.show()
