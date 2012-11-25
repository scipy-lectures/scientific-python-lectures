import pylab as pl
import numpy as np

size = 256, 16
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = pl.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
pl.axes([0, 0, 1, 1], frameon=False)

pl.plot(np.arange(3), [0, 1, 0], color="blue", dashes=[12, 5], linewidth=8,
        dash_joinstyle='miter')
pl.plot(4 + np.arange(3), [0, 1, 0], color="blue", dashes=[12, 5],
        linewidth=8, dash_joinstyle='bevel')
pl.plot(8 + np.arange(3), [0, 1, 0], color="blue", dashes=[12, 5],
        linewidth=8, dash_joinstyle='round')

pl.xlim(0, 12)
pl.ylim(-1, 2)
pl.xticks(())
pl.yticks(())

pl.show()
