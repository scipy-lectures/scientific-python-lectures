import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt


def plot_spectrum(F):
    plt.imshow(np.log(5 + np.abs(F)))


# read image
im = plt.imread('../../data/moonlanding.png').astype(float)

# Compute the 2d FFT of the input image
F = fftpack.fft2(im)

# In the lines following, we'll make a copy of the original spectrum and
# truncate coefficients.

# Define the fraction of coefficients (in each direction) we keep
keep_fraction = 0.1

# Call ff a copy of the original transform. Numpy arrays have a copy
# method for this purpose.
ff = F.copy()

# Set r and c to be the number of rows and columns of the array.
r, c = ff.shape

# Set to zero all rows with indices between r*keep_fraction and # r*(1-keep_fraction):
ff[r*keep_fraction:r*(1-keep_fraction)] = 0

# Similarly with the columns:
ff[:, c*keep_fraction:c*(1-keep_fraction)] = 0

# Reconstruct the denoised image from the filtered spectrum, keep only the
# real part for display.
im_new = fftpack.ifft2(ff).real

# Show the results
plt.figure(figsize=(12,8))
plt.subplot(221)
plt.title('Original image')
plt.imshow(im, plt.cm.gray)
plt.subplot(222)
plt.title('Fourier transform')
plot_spectrum(F)
plt.subplot(224)
plt.title('Filtered Spectrum')
plot_spectrum(ff)
plt.subplot(223)
plt.title('Reconstructed Image')
plt.imshow(im_new, plt.cm.gray)

# Adjust the spacing between subplots for readability
plt.subplots_adjust(hspace=0.4)

plt.show()
