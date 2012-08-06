"""
Smooth vs non-smooth
"""
import numpy as np
import pylab as pl

x = np.linspace(-1.5, 1.5, 101)

# A smooth function
pl.figure(1, figsize=(3, 2.5))
pl.clf()

pl.plot(x, np.sqrt(.2 + x**2), linewidth=2)
pl.text(-1, 0, '$f$', size=20)

pl.ylim(ymin=-.2)
pl.axis('off')
pl.tight_layout()

# A non-smooth function
pl.figure(2, figsize=(3, 2.5))
pl.clf()
pl.plot(x, np.abs(x), linewidth=2)
pl.text(-1, 0, '$f$', size=20)

pl.ylim(ymin=-.2)
pl.axis('off')
pl.tight_layout()
pl.show()

