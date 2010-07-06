from scipy import stats
import numpy as np
import pylab as pl
a = np.random.normal(size=10000)
bins = np.linspace(-5, 5, 30)
histogram, bins = np.histogram(a, bins=bins, normed=True)
bins = 0.5*(bins[1:] + bins[:-1])
from scipy import stats
b = stats.norm.pdf(bins)
pl.plot(bins, histogram)
pl.plot(bins, b)

