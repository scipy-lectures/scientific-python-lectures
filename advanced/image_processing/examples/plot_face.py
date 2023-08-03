"""
Displaying a Raccoon Face
=========================

Small example to plot a raccoon face.
"""

import scipy as sp
import imageio.v3 as iio

f = sp.datasets.face()
iio.imwrite("face.png", f)  # uses the Image module (PIL)

import matplotlib.pyplot as plt

plt.imshow(f)
plt.show()
