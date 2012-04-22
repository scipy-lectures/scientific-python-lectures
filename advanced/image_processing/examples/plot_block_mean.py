import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

l = scipy.misc.lena()
sx, sy = l.shape
X, Y = np.ogrid[0:sx, 0:sy]

regions = sy/6 * (X/4) + Y/6
block_mean = ndimage.mean(l, labels=regions,
                          index=np.arange(1, regions.max() +1))
block_mean.shape = (sx/4, sy/6)

plt.figure(figsize=(5, 5))
plt.imshow(block_mean, cmap=plt.cm.gray)
plt.axis('off')

plt.show()

