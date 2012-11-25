import pylab as pl
import numpy as np

pl.figure(figsize=(8, 5), dpi=80)
pl.subplot(111)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C = np.cos(X)
S = np.sin(X)

pl.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
pl.plot(X, S, color="red", linewidth=2.5, linestyle="-")

pl.xlim(X.min() * 1.1, X.max() * 1.1)
pl.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

pl.ylim(C.min() * 1.1, C.max() * 1.1)
pl.yticks([-1, 0, +1],
          [r'$-1$', r'$0$', r'$+1$'])

pl.show()
