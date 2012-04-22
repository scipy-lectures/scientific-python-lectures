import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

lena = scipy.misc.lena()
lx, ly = lena.shape
# Copping
crop_lena = lena[lx/4:-lx/4, ly/4:-ly/4]
# up <-> down flip
flip_ud_lena = np.flipud(lena)
# rotation
rotate_lena = ndimage.rotate(lena, 45)
rotate_lena_noreshape = ndimage.rotate(lena, 45, reshape=False)

plt.figure(figsize=(12.5, 2.5))


plt.subplot(151)
plt.imshow(lena, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(152)
plt.imshow(crop_lena, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(153)
plt.imshow(flip_ud_lena, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(154)
plt.imshow(rotate_lena, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(155)
plt.imshow(rotate_lena_noreshape, cmap=plt.cm.gray)
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0,
                    right=1)

plt.show()
