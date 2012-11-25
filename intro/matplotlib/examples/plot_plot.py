import pylab as pl
import numpy as np

n = 256
X = np.linspace(0, 2, n)
Y = np.sin(2 * np.pi * X)

pl.plot (X, Y, lw=2, color='violet')
pl.xlim(-0.2, 2.2)
pl.xticks(())
pl.ylim(-1.2, 1.2)
pl.yticks(())

pl.text(-0.05, 1.02, " Regular Plot:      pl.plot(...)\n",
        horizontalalignment='left',
        verticalalignment='top',
        size='xx-large',
        bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
        transform=pl.gca().transAxes)

pl.text(-0.05, 1.01, "\n\n   Plot lines and/or markers ",
        horizontalalignment='left',
        verticalalignment='top',
        size='large',
        transform=pl.gca().transAxes)

pl.show()
