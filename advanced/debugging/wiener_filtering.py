""" Wiener filtering a noisy Lena: this module is buggy
"""

import numpy as np
import scipy as sp
import pylab as pl
from scipy import signal


def local_mean(img, size=3):
    """ Compute a image of the local average
    """
    structure_element = np.ones((size, size), dtype=img.dtype)
    l_mean = signal.correlate(img, structure_element, mode='same')
    l_mean /= size**2
    return l_mean


def local_var(img, size=3):
    """ Compute a image of the local variance
    """
    structure_element = np.ones((size, size), dtype=img.dtype)
    l_var = signal.correlate(img**2, structure_element, mode='same')
    l_var /= size**2
    l_var -= local_mean(img, size=size)**2
    return l_var


def iterated_wiener(noisy_img, size=3):
    """ Wiener filter with iterative computation of the noise variance.

        Do not use this: this is crappy code to demo bugs!
    """
    noisy_img = noisy_img
    denoised_img = local_mean(noisy_img, size=size)
    l_var = local_var(noisy_img, size=size)
    for i in range(3):
        res = noisy_img - denoised_img
        noise = (res**2).sum()/res.size
        noise_level = (1 - noise/l_var )
        noise_level[noise_level<0] = 0
        denoised_img += noise_level*res
    return denoised_img


################################################################################
cut = (slice(128, -128), slice(128, -128))

np.random.seed(7)

lena = sp.lena()
noisy_lena = lena + 20*np.random.randint(3, size=lena.shape) - 30

pl.matshow(lena[cut], cmap=pl.cm.gray)
pl.matshow(noisy_lena[cut], cmap=pl.cm.gray)

denoised_lena = iterated_wiener(noisy_lena)
pl.matshow(denoised_lena[cut], cmap=pl.cm.gray)

pl.show()

