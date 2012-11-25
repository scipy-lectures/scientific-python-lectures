import pylab as pl
from mpl_toolkits.mplot3d import axes3d

ax = pl.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contourf(X, Y, Z)
ax.clabel(cset, fontsize=9, inline=1)

pl.xticks(())
pl.yticks(())
ax.set_zticks(())

ax.text2D(-0.05, 1.05, " 3D plots \n\n",
          horizontalalignment='left',
          verticalalignment='top',
          family='Lint McCree Intl BB',
          size='x-large',
          bbox=dict(facecolor='white', alpha=1.0, width=350,height=60),
          transform=pl.gca().transAxes)

ax.text2D(-0.05, .975, " Plot 2D or 3D data",
          horizontalalignment='left',
          verticalalignment='top',
          family='Lint McCree Intl BB',
          size='medium',
          transform=pl.gca().transAxes)

pl.show()
