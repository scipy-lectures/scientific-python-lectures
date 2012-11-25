import pylab as pl

ax = pl.axes([0.025, 0.025, 0.95, 0.95])

ax.set_xlim(0,4)
ax.set_ylim(0,3)
ax.xaxis.set_major_locator(pl.MultipleLocator(1.0))
ax.xaxis.set_minor_locator(pl.MultipleLocator(0.1))
ax.yaxis.set_major_locator(pl.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(pl.MultipleLocator(0.1))
ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
ax.set_xticklabels([])
ax.set_yticklabels([])

pl.show()
