import pylab as pl

size = 128, 16
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = pl.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)

pl.axes([0, 0, 1, 1], frameon=False)

pl.rcParams['text.antialiased'] = False
pl.text(0.5, 0.5, "Aliased", ha='center', va='center')

pl.xlim(0, 1)
pl.ylim(0, 1)
pl.xticks(())
pl.yticks(())

pl.show()
