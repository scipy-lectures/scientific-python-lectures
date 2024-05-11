"""
Displaying a Raccoon Face
=========================

Small example to plot a raccoon face.
"""

import imageio.v3 as iio
import scipy as sp

f = sp.datasets.face()
iio.imwrite("face.png", f)  # uses the Image module (PIL)

import matplotlib.pyplot as plt

plt.imshow(f)
plt.show()
