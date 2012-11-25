import pylab as pl
import numpy as np

pl.figure(figsize=(8,5), dpi=80)
pl.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C = np.cos(X)
S = np.sin(X)

pl.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
pl.plot(X, S, color="red", linewidth=2.5, linestyle="-",  label="sine")

ax = pl.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

pl.xlim(X.min() * 1.1, X.max() * 1.1)
pl.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

pl.ylim(C.min() * 1.1, C.max() * 1.1)
pl.yticks([-1, +1],
          [r'$-1$', r'$+1$'])

pl.legend(loc='upper left')

pl.show()
