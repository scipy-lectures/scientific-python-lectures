import pylab as pl

pl.figure(figsize=(6, 4))
pl.subplot(2, 2, 1)
pl.xticks(())
pl.yticks(())
pl.text(0.5, 0.5, 'subplot(2,2,1)', ha='center', va='center',
        size=20, alpha=.5)

pl.subplot(2, 2, 2)
pl.xticks(())
pl.yticks(())
pl.text(0.5, 0.5, 'subplot(2,2,2)', ha='center', va='center',
        size=20, alpha=.5)

pl.subplot(2, 2, 3)
pl.xticks(())
pl.yticks(())

pl.text(0.5, 0.5, 'subplot(2,2,3)', ha='center', va='center',
        size=20, alpha=.5)

pl.subplot(2, 2, 4)
pl.xticks(())
pl.yticks(())
pl.text(0.5, 0.5, 'subplot(2,2,4)', ha='center', va='center',
        size=20, alpha=.5)

pl.tight_layout()
pl.show()
