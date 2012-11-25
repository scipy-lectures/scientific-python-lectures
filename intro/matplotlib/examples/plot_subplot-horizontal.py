import pylab as pl

pl.figure(figsize=(6, 4))
pl.subplot(2, 1, 1)
pl.xticks(())
pl.yticks(())
pl.text(0.5, 0.5, 'subplot(2,1,1)', ha='center', va='center',
        size=24, alpha=.5)

pl.subplot(2, 1, 2)
pl.xticks(())
pl.yticks(())
pl.text(0.5, 0.5, 'subplot(2,1,2)', ha='center', va='center',
        size=24, alpha=.5)

pl.tight_layout()
pl.show()
