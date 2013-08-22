"""
A small demo of data animation
"""
import numpy as np
from mayavi import mlab

# Create some simple data
x , y , z = np.ogrid[-5:5:100j ,-5:5:100j, -5:5:100j]
scalars = np.sin(x * y * z) / (x * y * z)

iso = mlab.contour3d(scalars, transparent=True, contours=[0.5])
for i in range(1, 20):
    scalars = np.sin(i * x * y * z) /(x * y * z)
    iso.mlab_source.scalars = scalars

# Start the event loop, if needed
mlab.show()
