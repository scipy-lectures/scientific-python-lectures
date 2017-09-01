"""
Reading and writing an elephant
===============================

Read and write images

"""
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
img = plt.imread('../data/elephant.png')
plt.imshow(img)

plt.figure()
img_red = img[:, :, 0]
plt.imshow(img_red, cmap=plt.cm.gray)

plt.figure()
img_tiny = img[::6, ::6]
plt.imshow(img_tiny, interpolation='nearest') 

plt.show()
