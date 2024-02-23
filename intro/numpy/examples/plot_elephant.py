"""
Reading and writing an elephant
===============================

Read and write images

"""

import numpy as np
import matplotlib.pyplot as plt

#################################
# original figure
#################################

plt.figure()
img = plt.imread("../../../data/elephant.png")
plt.imshow(img)

#################################
# red channel displayed in grey
#################################

plt.figure()
img_red = img[:, :, 0]
plt.imshow(img_red, cmap=plt.cm.gray)

#################################
# lower resolution
#################################

plt.figure()
img_tiny = img[::6, ::6]
plt.imshow(img_tiny, interpolation="nearest")

plt.show()
