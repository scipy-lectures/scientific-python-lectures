"""
Grid elaborate
===============

An example displaying a grid on the axes and tweaking the layout.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
axes = plt.subplot(111)
axes.set_xlim(0, 4)
axes.set_ylim(0, 3)

axes.xaxis.set_major_locator(MultipleLocator(1.0))
axes.xaxis.set_minor_locator(MultipleLocator(0.1))
axes.yaxis.set_major_locator(MultipleLocator(1.0))
axes.yaxis.set_minor_locator(MultipleLocator(0.1))
axes.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
axes.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
axes.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
axes.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
axes.set_xticklabels([])
axes.set_yticklabels([])

plt.text(-0.05, 1.02, " Grid:                  plt.grid(...)\n",
          horizontalalignment='left',
          verticalalignment='top',
          size='xx-large',
          bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
          transform=axes.transAxes)

plt.text(-0.05, 1.01, "\n\n    Draw ticks and grid ",
          horizontalalignment='left',
          verticalalignment='top',
          size='large',
          transform=axes.transAxes)

