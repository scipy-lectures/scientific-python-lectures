import pylab as pl
import numpy as np

n = 20
X = np.ones(n)
X[-1] *= 2
pl.pie(X, explode=X*.05, colors = ['%f' % (i/float(n)) for i in range(n)])

fig = pl.gcf()
w, h = fig.get_figwidth(), fig.get_figheight()
r = h / float(w)

pl.xlim(-1.5, 1.5)
pl.ylim(-1.5 * r, 1.5 * r)
pl.xticks(())
pl.yticks(())

pl.text(-0.05, 1.02, " Pie Chart:           pl.pie(...)\n",
        horizontalalignment='left',
        verticalalignment='top',
        size='xx-large',
        bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
        transform=pl.gca().transAxes)

pl.text(-0.05, 1.01, "\n\n   Make a pie chart of an array ",
        horizontalalignment='left',
        verticalalignment='top',
        size='large',
        transform=pl.gca().transAxes)

pl.show()
