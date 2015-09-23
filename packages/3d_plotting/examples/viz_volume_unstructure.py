"""
Visualizing unstructured data
==============================

Use Mayavi to visualize unstructured data
"""

from mayavi import mlab
import numpy as np

np.random.seed(0)

x, y, z = 10 * np.random.random(size=(3, 200))

data = x*x*0.5 + y*y + z*z*2.0

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.clf()

src = mlab.pipeline.scalar_scatter(x, y, z, data)
mlab.pipeline.glyph(src, mode='cube', scale_factor=0.4, scale_mode='none')
mlab.savefig('viz_volume_unstructure.png')
mlab.show()


