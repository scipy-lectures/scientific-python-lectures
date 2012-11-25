import pylab as pl

fig = pl.figure()
fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

pl.subplot(2, 1, 1)
pl.xticks(()), pl.yticks(())

pl.subplot(2, 3, 4)
pl.xticks(())
pl.yticks(())

pl.subplot(2, 3, 5)
pl.xticks(())
pl.yticks(())

pl.subplot(2, 3, 6)
pl.xticks(())
pl.yticks(())

pl.show()
