"""
Alpha: transparency
===================

This example demonstrates using alpha for transparency.
"""

import matplotlib.pyplot as plt

size = 256,16
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, 0.1, 1, .8], frameon=False)

for i in range(1, 11):
    plt.axvline(i, linewidth=1, color='blue', alpha= .25 + .75 * i / 10.)

plt.xlim(0, 11)
plt.xticks(())
plt.yticks(())
plt.show()
