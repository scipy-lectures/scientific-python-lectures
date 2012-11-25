import pylab as pl
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

pl.axes([0.025, 0.025, 0.95, 0.95])
pl.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
pl.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    pl.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va= 'bottom')

for x, y in zip(X, Y2):
    pl.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va= 'top')

pl.xlim(-.5, n)
pl.xticks(())
pl.ylim(-1.25, 1.25)
pl.yticks(())

pl.show()
