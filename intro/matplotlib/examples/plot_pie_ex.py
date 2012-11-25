import pylab as pl
import numpy as np

n = 20
Z = np.ones(n)
Z[-1] *= 2

pl.axes([0.025, 0.025, 0.95, 0.95])

pl.pie(Z, explode=Z*.05, colors = ['%f' % (i/float(n)) for i in range(n)])
pl.axis('equal')
pl.xticks(())
pl.yticks()

pl.show()
