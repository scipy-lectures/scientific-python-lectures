import numpy as np

x, y = np.mgrid[-10:10:100j, -10:10:100j]
r = np.sqrt(x**2 + y**2)
z = np.sin(r)/r

from enthought.mayavi import mlab
mlab.surf(z, warp_scale='auto')

mlab.outline()
mlab.axes()
