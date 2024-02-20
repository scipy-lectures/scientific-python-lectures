"""
Total Variation denoising
===========================

This example demoes Total-Variation (TV) denoising on a Raccoon face.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from skimage.restoration import denoise_tv_chambolle

rng = np.random.default_rng(27446968)

f = sp.datasets.face(gray=True)
f = f[230:290, 220:320]

noisy = f + 0.4 * f.std() * rng.random(f.shape)

tv_denoised = denoise_tv_chambolle(noisy, weight=10)


plt.figure(figsize=(12, 2.8))

plt.subplot(131)
plt.imshow(noisy, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis("off")
plt.title("noisy", fontsize=20)
plt.subplot(132)
plt.imshow(tv_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis("off")
plt.title("TV denoising", fontsize=20)

tv_denoised = denoise_tv_chambolle(noisy, weight=50)
plt.subplot(133)
plt.imshow(tv_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
plt.axis("off")
plt.title("(more) TV denoising", fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0, right=1)
plt.show()
