import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('../data/elephant.png')
print img.shape, img.dtype
# (200, 300, 3)  dtype('float32')

plt.imshow(img)
plt.savefig('plot.png')
plt.show()

plt.imsave('red_elephant', img[:,:,0], cmap=plt.cm.gray)

# This saved only one channel (of RGB)

plt.imshow(plt.imread('red_elephant.png'))
plt.show()

# Other libraries:

from scipy.misc import imsave
imsave('tiny_elephant.png', img[::6,::6])
plt.imshow(plt.imread('tiny_elephant.png'), interpolation='nearest')
plt.show()
