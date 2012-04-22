import numpy as np
import scipy
import matplotlib.pyplot as plt

lena = scipy.misc.lena()
lena[10:13, 20:23]
lena[100:120] = 255

lx, ly = lena.shape
X, Y = np.ogrid[0:lx, 0:ly]
mask = (X - lx/2)**2 + (Y - ly/2)**2 > lx*ly/4
lena[mask] = 0
lena[range(400), range(400)] = 255

plt.figure(figsize=(3, 3))
plt.axes([0, 0, 1, 1])
plt.imshow(lena, cmap=plt.cm.gray)
plt.axis('off')

plt.show()
