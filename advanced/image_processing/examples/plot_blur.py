import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

lena = scipy.misc.lena()
blurred_lena = ndimage.gaussian_filter(lena, sigma=3)
very_blurred = ndimage.gaussian_filter(lena, sigma=5)
local_mean = ndimage.uniform_filter(lena, size=11)

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.imshow(blurred_lena, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(132)
plt.imshow(very_blurred, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(local_mean, cmap=plt.cm.gray)
plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01,
                    left=0.01, right=0.99)

plt.show()
