"""
Linestyles
==========

Plot the different line styles.
"""

import numpy as np
import matplotlib.pyplot as plt

def linestyle(ls, i):
    X = i * .5 * np.ones(11)
    Y = np.arange(11)
    plt.plot(X, Y, ls, color=(.0, .0, 1, 1), lw=3, ms=8,
            mfc=(.75, .75, 1, 1), mec=(0, 0, 1, 1))
    plt.text(.5 * i, 10.25, ls, rotation=90, fontsize=15, va='bottom')

linestyles = ['-', '--', ':', '-.', '.', ',', 'o', '^', 'v', '<', '>', 's',
              '+', 'x', 'd', '1', '2', '3', '4', 'h', 'p', '|', '_', 'D', 'H']
n_lines = len(linestyles)

size = 20 * n_lines, 300
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi)
plt.axes([0, 0.01, 1, .9], frameon=False)

for i, ls in enumerate(linestyles):
    linestyle(ls, i)

plt.xlim(-.2, .2 + .5*n_lines)
plt.xticks(())
plt.yticks(())

plt.show()
