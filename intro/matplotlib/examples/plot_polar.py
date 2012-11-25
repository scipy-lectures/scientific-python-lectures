import pylab as pl
import numpy as np

pl.subplot(1, 1, 1, polar=True)

N = 20
theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
bars = pl.bar(theta, radii, width=width, bottom=0.0)
for r, bar in zip(radii, bars):
    bar.set_facecolor(pl.cm.jet(r / 10.))
    bar.set_alpha(0.5)
pl.gca().set_xticklabels([])
pl.gca().set_yticklabels([])

pl.text(-0.2, 1.02, " Polar Axis\n",
      horizontalalignment='left',
      verticalalignment='top',
      size='xx-large',
      bbox=dict(facecolor='white', alpha=1.0, width=400, height=65),
      transform=pl.gca().transAxes)
pl.text(-0.2, 1.01, "\n\n Plot anything using polar axis ",
      horizontalalignment='left',
      verticalalignment='top',
      size='large',
      transform=pl.gca().transAxes)

pl.show()
