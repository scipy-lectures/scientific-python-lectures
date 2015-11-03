"""
Image interpolation
=====================

The example demonstrates image interpolation on a face.
"""

import scipy.misc
import matplotlib.pyplot as plt

f = scipy.misc.face(gray=True)

plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.imshow(f[200:220, 200:220], cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(122)
plt.imshow(f[200:220, 200:220], cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
plt.show()
