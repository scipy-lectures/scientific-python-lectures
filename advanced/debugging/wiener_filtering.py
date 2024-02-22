""" Wiener filtering a noisy raccoon face: this module is buggy
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def local_mean(img, size=3):
    """Compute a image of the local average"""
    structure_element = np.ones((size, size), dtype=img.dtype)
    l_mean = sp.signal.correlate(img, structure_element, mode="same")
    l_mean /= size**2
    return l_mean


def local_var(img, size=3):
    """Compute a image of the local variance"""
    structure_element = np.ones((size, size), dtype=img.dtype)
    l_var = sp.signal.correlate(img**2, structure_element, mode="same")
    l_var /= size**2
    l_var -= local_mean(img, size=size) ** 2
    return l_var


def iterated_wiener(noisy_img, size=3):
    """Wiener filter with iterative computation of the noise variance.

    Do not use this: this is crappy code to demo bugs!
    """
    noisy_img = noisy_img
    denoised_img = local_mean(noisy_img, size=size)
    l_var = local_var(noisy_img, size=size)
    for i in range(3):
        res = noisy_img - denoised_img
        noise = (res**2).sum() / res.size
        noise_level = 1 - noise / l_var
        noise_level[noise_level < 0] = 0
        denoised_img += noise_level * res
    return denoised_img


################################################################################
cut = (slice(128, -128), slice(128, -128))

rng = np.random.default_rng(27446968)

face = sp.datasets.face(gray=True)
noisy_face = face + 20 * rng.integers(3, size=face.shape) - 30

plt.matshow(face[cut], cmap=plt.cm.gray)
plt.matshow(noisy_face[cut], cmap=plt.cm.gray)

denoised_face = iterated_wiener(noisy_face)
plt.matshow(denoised_face[cut], cmap=plt.cm.gray)

plt.show()
