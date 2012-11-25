import pylab as pl
import numpy as np

pl.figure(figsize=(8, 5), dpi=80)
pl.subplot(111)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

pl.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
pl.plot(X, S, color="red", linewidth=2.5, linestyle="-")

pl.xlim(-4.0, 4.0)
pl.xticks(np.linspace(-4, 4, 9, endpoint=True))

pl.ylim(-1.0, 1.0)
pl.yticks(np.linspace(-1, 1, 5, endpoint=True))

pl.show()
