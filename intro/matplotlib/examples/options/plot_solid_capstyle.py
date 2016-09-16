"""
Solid cap style
================

An example demoing the solide cap style in matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

size = 256, 16
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, 0, 1, 1], frameon=False)

plt.plot(np.arange(4), np.ones(4), color="blue", linewidth=8,
        solid_capstyle='butt')

plt.plot(5 + np.arange(4), np.ones(4), color="blue", linewidth=8,
        solid_capstyle='round')

plt.plot(10 + np.arange(4), np.ones(4), color="blue", linewidth=8,
        solid_capstyle='projecting')

plt.xlim(0, 14)
plt.xticks(())
plt.yticks(())

plt.show()
