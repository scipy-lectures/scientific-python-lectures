"""
Definition of a convex function
"""

import numpy as np
import pylab as pl

x = np.linspace(-1, 2)

pl.figure(1, figsize=(3, 2.5))
pl.clf()

# A convex function
pl.plot(x, x**2, linewidth=2)
pl.text(-.7, -.6**2, '$f$', size=20)

# The tangent in one point
pl.plot(x, 2*x - 1)
pl.plot(1, 1, 'k+')
pl.text(.3, -.75, "Tangent to $f$", size=15)
pl.text(1, 1 - .5, 'C', size=15)

# Convexity as barycenter
pl.plot([.35, 1.85], [.35**2, 1.85**2])
pl.plot([.35, 1.85], [.35**2, 1.85**2], 'k+')
pl.text(.35 - .2, .35**2 + .1, 'A', size=15)
pl.text(1.85 - .2, 1.85**2, 'B', size=15)

pl.ylim(ymin=-1)
pl.axis('off')
pl.tight_layout()

# Convexity as barycenter
pl.figure(2, figsize=(3, 2.5))
pl.clf()
pl.plot(x, x**2 + np.exp(-5*(x - .5)**2), linewidth=2)
pl.text(-.7, -.6**2, '$f$', size=20)

pl.ylim(ymin=-1)
pl.axis('off')
pl.tight_layout()
pl.show()

