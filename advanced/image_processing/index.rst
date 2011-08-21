=======================================================
Image manipulation and processing using Numpy and Scipy
=======================================================

Introduction
============

Opening and writing to image files
==================================

Writing an array to a file ::

    >>> import scipy
    >>> l = scipy.lena()
    >>> from scipy import misc
    >>> misc.imsave('lena.png', l) # uses the Image module (PIL)

Creating a numpy array from an image file ::

    >>> lena = misc.imread('lena.png')
    >>> type(lena)
    <type 'numpy.ndarray'>
    >>> lena.shape, lena.dtype
    ((512, 512), dtype('uint8'))

dtype is uint8 for 8-bit images (0-255)

Opening raw files (camera, 3-D images) ::

    >>> l.tofile('lena.raw') # Create raw file
    >>> lena_from_raw = np.fromfile('lena.raw', dtype=np.int64)
    >>> lena_from_raw.shape
    (262144,)
    >>> lena_from_raw.shape = (512, 512)
    >>> import os
    >>> os.remove('lena.raw')

Need to know the shape and dtype of the image (how to separate data
bytes).

For large data, use ``np.memmap`` for memory mapping::

    >>> lena_memmap = np.memmap('lena.raw', dtype=np.int64, shape=(512, 512))

(data are read from the file, and not loaded into memory)

Working on a list of image files ::

    >>> from glob import glob
    >>> filelist = glob('pattern*.png')
    >>> filelist.sort()

Basic manipulations
===================

Images are arrays: use the whole ``numpy`` machinery.
