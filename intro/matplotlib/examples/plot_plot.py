"""
Plot example vignette
=======================

An example of plots with matplotlib, and added annotations.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 256
X = np.linspace(0, 2, n)
Y = np.sin(2 * np.pi * X)

plt.plot (X, Y, lw=2, color='violet')
plt.xlim(-0.2, 2.2)
plt.xticks(())
plt.ylim(-1.2, 1.2)
plt.yticks(())

plt.text(-0.05, 1.02, " Regular Plot:      plt.plot(...)\n",
        horizontalalignment='left',
        verticalalignment='top',
        size='xx-large',
        bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
        transform=plt.gca().transAxes)

plt.text(-0.05, 1.01, "\n\n   Plot lines and/or markers ",
        horizontalalignment='left',
        verticalalignment='top',
        size='large',
        transform=plt.gca().transAxes)

plt.show()
