import pylab as pl
import numpy as np

n = 256
X = np.linspace(-np.pi, np.pi, n, endpoint=True)
Y = np.sin(2 * X)

pl.axes([0.025, 0.025, 0.95, 0.95])

pl.plot(X, Y + 1, color='blue', alpha=1.00)
pl.fill_between(X, 1, Y + 1, color='blue', alpha=.25)

pl.plot(X, Y - 1, color='blue', alpha=1.00)
pl.fill_between(X, -1, Y - 1, (Y - 1) > -1, color='blue', alpha=.25)
pl.fill_between(X, -1, Y - 1, (Y - 1) < -1, color='red',  alpha=.25)

pl.xlim(-np.pi, np.pi)
pl.xticks(())
pl.ylim(-2.5, 2.5)
pl.yticks(())

pl.show()
