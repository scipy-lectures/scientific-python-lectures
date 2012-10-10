"""
Use Mayavi to visualize the structure of a VolumeImg
"""

from enthought.mayavi import mlab
import numpy as np

x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]

data = x*x*0.5 + y*y + z*z*2.0

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.clf()

src = mlab.pipeline.scalar_field(x, y, z, data)

mlab.pipeline.surface(src, opacity=0.4)

src2 = mlab.pipeline.scalar_field(x[::9, ::9, ::9],
                                  y[::9, ::9, ::9],
                                  z[::9, ::9, ::9],
                                  data[::9, ::9, ::9])
mlab.pipeline.surface(mlab.pipeline.extract_edges(src2), color=(0, 0, 0))
mlab.pipeline.glyph(src2, mode='cube', scale_factor=0.4, scale_mode='none')
mlab.savefig('viz_volume_structure.png')
mlab.show()


