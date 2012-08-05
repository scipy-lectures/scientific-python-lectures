"""
Noisy vs non-noisy
"""
import numpy as np
import pylab as pl

np.random.seed(0)

x = np.linspace(-5, 5, 101)
x_ = np.linspace(-5, 5, 31)

def f(x):
    return -np.exp(-x**2)

# A smooth function
pl.figure(1, figsize=(3, 2.5))
pl.clf()

pl.plot(x_, f(x_) + .2*np.random.normal(size=31), linewidth=2)
pl.plot(x, f(x), linewidth=2)

pl.ylim(ymin=-1.3)
pl.axis('off')
pl.tight_layout()
pl.show()

