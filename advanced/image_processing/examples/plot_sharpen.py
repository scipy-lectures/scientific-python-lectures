import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

l = scipy.misc.lena()
blurred_l = ndimage.gaussian_filter(l, 3)

filter_blurred_l = ndimage.gaussian_filter(blurred_l, 1)

alpha = 30
sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(l, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(132)
plt.imshow(blurred_l, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(sharpened, cmap=plt.cm.gray)
plt.axis('off')

plt.show()
