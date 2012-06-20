
.. only:: html


>>> import numpy as np
>>> import pylab as pl
>>> from scipy import ndimage

.. _image-answers:

Example of solution for the image processing exercise: unmolten grains in glass
===============================================================================


  .. image:: ../image_processing/MV_HFV_012.jpg
     :align: center
     :scale: 70

1. Open the image file MV_HFV_012.jpg and display it. Browse through the
   keyword arguments in the docstring of ``imshow`` to display the image
   with the "right" orientation (origin in the bottom left corner, and not
   the upper left corner as for standard arrays). ::

    >>> dat = pl.imread('data/MV_HFV_012.jpg')

2. Crop the image to remove the lower panel with measure information. ::

    >>> dat = dat[60:]

3. Slightly filter the image with a median filter in order to refine its
   histogram. Check how the histogram changes. ::

    >>> filtdat = ndimage.median_filter(dat, size=(7,7))
    >>> hi_dat = np.histogram(dat, bins=np.arange(256))
    >>> hi_filtdat = np.histogram(filtdat, bins=np.arange(256))

.. image:: ../image_processing/exo_histos.png
   :align: center

4. Using the histogram of the filtered image, determine thresholds that
   allow to define masks for sand pixels, glass pixels and bubble pixels.
   Other option (homework): write a function that determines automatically
   the thresholds from the minima of the histogram. ::

    >>> void = filtdat <= 50
    >>> sand = np.logical_and(filtdat > 50, filtdat <= 114)
    >>> glass = filtdat > 114

5. Display an image in which the three phases are colored with three
   different colors. ::

    >>> phases = void.astype(np.int) + 2*glass.astype(np.int) + 3*sand.astype(np.int)

.. image:: ../image_processing/three_phases.png
   :align: center

6. Use mathematical morphology to clean the different phases. ::

    >>> sand_op = ndimage.binary_opening(sand, iterations=2)

7. Attribute labels to all bubbles and sand grains, and remove from the
   sand mask grains that are smaller than 10 pixels. To do so, use
   ``ndimage.sum`` or ``np.bincount`` to compute the grain sizes. ::

    >>> sand_labels, sand_nb = ndimage.label(sand_op)
    >>> sand_areas = np.array(ndimage.sum(sand_op, sand_labels, np.arange(sand_labels.max()+1)))
    >>> mask = sand_areas > 100
    >>> remove_small_sand = mask[sand_labels.ravel()].reshape(sand_labels.shape)

.. image:: ../image_processing/sands.png
   :align: center


8. Compute the mean size of bubbles. ::

    >>> bubbles_labels, bubbles_nb = ndimage.label(void)
    >>> bubbles_areas = np.bincount(bubbles_labels.ravel())[1:]
    >>> mean_bubble_size = bubbles_areas.mean()
    >>> median_bubble_size = np.median(bubbles_areas)
    >>> mean_bubble_size, median_bubble_size
    (1699.875, 65.0)
