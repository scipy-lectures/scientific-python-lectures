import pylab as pl

size = 256, 16
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = pl.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
pl.axes([0, 0, 1, 1], frameon=False)

for i in range(1,11):
    pl.plot([i, ], [1, ], 's', markersize=5,
         markeredgewidth=1 + i/10., markeredgecolor='k', markerfacecolor='w')
pl.xlim(0, 11)
pl.xticks(())
pl.yticks(())

pl.show()
