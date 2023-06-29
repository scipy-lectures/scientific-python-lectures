"""
==========================================
Comparing 2 sets of samples from Gaussians
==========================================

"""

import numpy as np
import matplotlib.pyplot as plt

# Generates 2 sets of observations
rng = np.random.default_rng(27446968)
samples1 = rng.normal(0, size=1000)
samples2 = rng.normal(1, size=1000)

# Compute a histogram of the sample
bins = np.linspace(-4, 4, 30)
histogram1, bins = np.histogram(samples1, bins=bins, density=True)
histogram2, bins = np.histogram(samples2, bins=bins, density=True)

plt.figure(figsize=(6, 4))
plt.hist(samples1, bins=bins, density=True, label="Samples 1")
plt.hist(samples2, bins=bins, density=True, label="Samples 2")
plt.legend(loc="best")
plt.show()
