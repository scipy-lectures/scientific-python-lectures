"""
Visualize the field created by a pair of Helmoltz coils
"""

import numpy as np
from scipy import stats

from mayavi import mlab

# "import" our data from our previous script (the import actually runs
# the script and computes B)
from compute_field import B, X, Y, Z

###############################################################################
# Data massaging

# Reshape the data to put it in a form that can be fed in Mayavi
Bx = B[:, 0]
By = B[:, 1]
Bz = B[:, 2]
Bx = np.reshape(Bx, X.shape)
By = np.reshape(By, X.shape)
Bz = np.reshape(Bz, X.shape)

Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

# Threshold, to avoid the very high values
Bmax = stats.scoreatpercentile(Bnorm.ravel(), 99)

Bx[Bnorm > Bmax] = Bmax * (Bx / Bnorm)[Bnorm > Bmax]
By[Bnorm > Bmax] = Bmax * (By / Bnorm)[Bnorm > Bmax]
Bz[Bnorm > Bmax] = Bmax * (Bz / Bnorm)[Bnorm > Bmax]
Bnorm[Bnorm > Bmax] = Bmax

###############################################################################
# Visualization proper

# Create a mayavi figure black on white
mlab.figure(bgcolor=(0., 0., 0.), fgcolor=(1., 1., 1.), size=(640, 480))

# Create a vector_field: a data source that we can slice and dice
field = mlab.pipeline.vector_field(X, Y, Z, Bx, By, Bz,
                                   scalars=Bnorm,
                                   name='B field')
# Plot the vectors
vectors = mlab.pipeline.vectors(field,
                                scale_factor=abs(X[0, 0, 0] - X[1, 1, 1]),
                                colormap='hot')
mlab.axes()

# Mask 7 data points out of 8
vectors.glyph.mask_input_points = True
vectors.glyph.mask_points.on_ratio = 8

mlab.pipeline.vector_cut_plane(field, scale_factor=.1, colormap='hot')

# Add an iso_surface of the norm of the field
mlab.pipeline.iso_surface(mlab.pipeline.extract_vector_norm(field),
                          contours=[0.1*Bmax, 0.4*Bmax],
                          opacity=0.5, transparent=True)

mlab.view(28, 84, 0.71)
mlab.savefig('visualize_field.png')
