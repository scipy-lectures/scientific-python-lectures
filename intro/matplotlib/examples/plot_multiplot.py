import pylab as pl

ax = pl.subplot(2, 1, 1)
ax.set_xticklabels([])
ax.set_yticklabels([])

pl.text(-0.05, 1.02, " Multiplot:     pl.subplot(...)\n",
      horizontalalignment='left',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
      transform=ax.transAxes)
pl.text(-0.05, 1.01, "\n\n    Plot several plots at once ",
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      transform=ax.transAxes)

ax = pl.subplot(2, 2, 3)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax = pl.subplot(2, 2, 4)
ax.set_xticklabels([])
ax.set_yticklabels([])

pl.show()
