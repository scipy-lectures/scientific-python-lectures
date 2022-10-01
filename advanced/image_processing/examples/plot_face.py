"""
Displaying a Racoon Face
========================

Small example to plot a racoon face.
"""

from scipy import misc
import imageio.v3 as iio
f = misc.face()
iio.imwrite('face.png', f) # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(f)
plt.show()
