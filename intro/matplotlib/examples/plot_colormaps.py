import pylab as pl
import numpy as np

pl.rc('text', usetex=False)
a = np.outer(np.arange(0, 1, 0.01), np.ones(10))

pl.figure(figsize=(10, 5))
pl.subplots_adjust(top=0.8, bottom=0.05, left=0.01, right=0.99)
maps = [m for m in pl.cm.datad if not m.endswith("_r")]
maps.sort()
l = len(maps) + 1

for i, m in enumerate(maps):
    pl.subplot(1, l, i+1)
    pl.axis("off")
    pl.imshow(a, aspect='auto', cmap=pl.get_cmap(m), origin="lower")
    pl.title(m, rotation=90, fontsize=10, va='bottom')

pl.show()
