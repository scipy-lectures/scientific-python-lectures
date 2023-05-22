"""
=======================================
Normal distribution: histogram and PDF
=======================================

Explore the normal distribution: a histogram built from samples and the
PDF (probability density function).
"""

import numpy as np

# Sample from a normal distribution using NumPy's random number generator
samples = np.random.normal(size=10000)

# Compute a histogram of the sample
bins = np.linspace(-5, 5, 30)
histogram, bins = np.histogram(samples, bins=bins, density=True)

bin_centers = 0.5*(bins[1:] + bins[:-1])

# Compute the PDF on the bin centers from SciPy distribution object
import scipy as sp
pdf = sp.stats.norm.pdf(bin_centers)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, histogram, label="Histogram of samples")
plt.plot(bin_centers, pdf, label="PDF")
plt.legend()
plt.show()
