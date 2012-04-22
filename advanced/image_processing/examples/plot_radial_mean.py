import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

l = scipy.misc.lena()
sx, sy = l.shape
X, Y = np.ogrid[0:sx, 0:sy]


r = np.hypot(X - sx/2, Y - sy/2)

rbin = (20* r/r.max()).astype(np.int)
radial_mean = ndimage.mean(l, labels=rbin, index=np.arange(1, rbin.max() +1))

plt.figure(figsize=(5, 5))
plt.axes([0, 0, 1, 1])
plt.imshow(rbin, cmap=plt.cm.spectral)
plt.axis('off')

plt.show()
