import pylab as pl

size = 256,16
dpi = 72.0
figsize= size[0] / float(dpi), size[1] / float(dpi)
fig = pl.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)
pl.axes([0, 0.1, 1, .8], frameon=False)

for i in range(1, 11):
    pl.axvline(i, linewidth=1, color='blue', alpha= .25 + .75 * i / 10.)

pl.xlim(0, 11)
pl.xticks(())
pl.yticks(())
pl.show()
