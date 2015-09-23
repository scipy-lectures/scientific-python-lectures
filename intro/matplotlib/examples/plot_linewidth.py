"""
Linewidth
=========

Plot various linewidth with matplotlib.
"""

import matplotlib.pyplot as plt

size = 256, 16
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
plt.axes([0, .1, 1, .8], frameon=False)

for i in range(1, 11):
    plt.plot([i, i], [0, 1], color='b', lw=i/2.)

plt.xlim(0, 11)
plt.ylim(0, 1)
plt.xticks(())
plt.yticks(())

plt.show()
