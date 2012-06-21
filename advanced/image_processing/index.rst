.. for doctests
   >>> import numpy as np
   >>> np.random.seed(0)


=======================================================
Image manipulation and processing using Numpy and Scipy
=======================================================

:authors: Emmanuelle Gouillart, Gaël Varoquaux


.. topic:: 
    Image = 2-D numerical array 

    (or 3-D: CT, MRI, 2D + time; 4-D, ...)

    Here, **image == Numpy array** ``np.array``

**Tools used in this tutorial**:

* ``numpy``: basic array manipulation

* ``scipy``: ``scipy.ndimage`` submodule dedicated to image processing 
  (n-dimensional images). See http://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html ::

    >>> from scipy import ndimage

* a few examples use specialized toolkits working with ``np.array``:

    * `Scikit Image <http://scikits-image.org/>`_
    
    * `scikit-learn <http://scikit-learn.org/>`_ 

**Common tasks in image processing**:

* Input/Output, displaying images

* Basic manipulations: cropping, flipping, rotating, ...

* Image filtering: denoising, sharpening

* Image segmentation: labeling pixels corresponding to different objects

* Classification

* ...


More powerful and complete modules:

* `OpenCV <http://opencv.willowgarage.com/documentation/python/cookbook.html>`_ 
  (Python bindings)

* `CellProfiler <http://www.cellprofiler.org>`_

* `ITK <http://www.itk.org/>`_ with Python bindings

* many more...

.. contents:: Chapters contents
   :local:
   :depth: 4



Opening and writing to image files
==================================

Writing an array to a file:

.. literalinclude:: examples/plot_lena.py
   :lines: 2-

.. image:: examples/lena.png
    :align: center
    :scale: 50


Creating a numpy array from an image file::

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

    >>> for i in range(10):
    ...     im = np.random.random_integers(0, 255, 10000).reshape((100, 100))
    ...     misc.imsave('random_%02d.png' % i, im)
    >>> from glob import glob
    >>> filelist = glob('random*.png')
    >>> filelist.sort()

Displaying images
=================

Use ``matplotlib`` and ``imshow`` to display an image inside a
``matplotlib figure``::

    >>> l = scipy.lena()
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(l, cmap=plt.cm.gray)
    <matplotlib.image.AxesImage object at 0x3c7f710>

Increase contrast by setting min and max values::

    >>> plt.imshow(l, cmap=plt.cm.gray, vmin=30, vmax=200)
    <matplotlib.image.AxesImage object at 0x33ef750>
    >>> # Remove axes and ticks
    >>> plt.axis('off')
    (-0.5, 511.5, 511.5, -0.5)

Draw contour lines::

    >>> plt.contour(l, [60, 211])
    <matplotlib.contour.ContourSet instance at 0x33f8c20>


.. figure:: auto_examples/images/plot_display_lena_1.png
    :scale: 100
    :target: auto_examples/plot_display_lena.html

.. only:: html

    [:ref:`Python source code <example_plot_display_lena.py>`]

For fine inspection of intensity variations, use
``interpolation='nearest'``::

    >>> plt.imshow(l[200:220, 200:220], cmap=plt.cm.gray)
    >>> plt.imshow(l[200:220, 200:220], cmap=plt.cm.gray, interpolation='nearest')

.. figure:: auto_examples/images/plot_interpolation_lena_1.png
    :scale: 80
    :target: auto_examples/plot_interpolation_lena.html

.. only:: html

    [:ref:`Python source code <example_plot_interpolation_lena.py>`]

Other packages sometimes use graphical toolkits for visualization (GTK,
Qt)::

    >>> import scikits.image.io as im_io
    >>> im_io.use_plugin('gtk', 'imshow')
    >>> im_io.imshow(l)

.. topic:: 3-D visualization: Mayavi

    See :ref:`mayavi-label` and :ref:`mayavi-voldata-label`.
    
	* Image plane widgets

	* Isosurfaces

	* ...

    .. image:: ../3d_plotting/ipw.png
	:align: center
	:scale: 65


Basic manipulations
===================

Images are arrays: use the whole ``numpy`` machinery.

.. image:: axis_convention.png
    :align: center
    :scale: 65

::

    >>> lena = scipy.lena()
    >>> lena[0, 40]
    166
    >>> # Slicing
    >>> lena[10:13, 20:23]
    array([[158, 156, 157],
    [157, 155, 155],
    [157, 157, 158]])
    >>> lena[100:120] = 255
    >>> 
    >>> lx, ly = lena.shape
    >>> X, Y = np.ogrid[0:lx, 0:ly]
    >>> mask = (X - lx/2)**2 + (Y - ly/2)**2 > lx*ly/4
    >>> # Masks
    >>> lena[mask] = 0
    >>> # Fancy indexing
    >>> lena[range(400), range(400)] = 255

.. figure:: auto_examples/images/plot_numpy_array_1.png
    :scale: 100
    :target: auto_examples/plot_numpy_array.html

.. only:: html

    [:ref:`Python source code <example_plot_numpy_array.py>`]

Statistical information
-----------------------

::

    >>> lena = scipy.lena()
    >>> lena.mean()
    124.04678344726562
    >>> lena.max(), lena.min()
    (245, 25)


``np.histogram``

Geometrical transformations
---------------------------
::

    >>> lena = scipy.lena()
    >>> lx, ly = lena.shape
    >>> # Cropping
    >>> crop_lena = lena[lx/4:-lx/4, ly/4:-ly/4]
    >>> # up <-> down flip
    >>> flip_ud_lena = np.flipud(lena)
    >>> # rotation
    >>> rotate_lena = ndimage.rotate(lena, 45)
    >>> rotate_lena_noreshape = ndimage.rotate(lena, 45, reshape=False)

.. figure:: auto_examples/images/plot_geom_lena_1.png
    :scale: 80
    :target: auto_examples/plot_geom_lena.html

.. only:: html

    [:ref:`Python source code <example_plot_geom_lena.py>`]

Image filtering
===============

**Local filters**: replace the value of pixels by a function of the values of
neighboring pixels. 

Neighbourhood: square (choose size), disk, or more complicated *structuring
element*.

.. image:: kernels.png
    :align: center

Blurring/smoothing
------------------

**Gaussian filter** from ``scipy.ndimage``::

    >>> from scipy import misc
    >>> lena = misc.lena()
    >>> blurred_lena = ndimage.gaussian_filter(lena, sigma=3)
    >>> very_blurred = ndimage.gaussian_filter(lena, sigma=5)

**Uniform filter** ::

    >>> local_mean = ndimage.uniform_filter(lena, size=11)

.. figure:: auto_examples/images/plot_blur_1.png
    :scale: 80
    :target: auto_examples/plot_blur.html

.. only:: html

    [:ref:`Python source code <example_plot_blur.py>`]

Sharpening
----------

Sharpen a blurred image::

    >>> from scipy import misc
    >>> lena = misc.lena()
    >>> blurred_l = ndimage.gaussian_filter(lena, 3)

increase the weight of edges by adding an approximation of the
Laplacian::

    >>> filter_blurred_l = ndimage.gaussian_filter(blurred_l, 1)
    >>> alpha = 30
    >>> sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)

.. figure:: auto_examples/images/plot_sharpen_1.png
    :scale: 100
    :target: auto_examples/plot_sharpen.html

.. only:: html

    [:ref:`Python source code <example_plot_sharpen.py>`]


Denoising
---------

Noisy lena::

    >>> from scipy import misc
    >>> l = misc.lena()
    >>> l = l[230:310, 210:350]
    >>> noisy = l + 0.4*l.std()*np.random.random(l.shape)

A **Gaussian filter** smoothes the noise out... and the edges as well::

    >>> gauss_denoised = ndimage.gaussian_filter(noisy, 2)

Most local linear isotropic filters blur the image (``ndimage.uniform_filter``)

A **median filter** preserves better the edges::

    >>> med_denoised = ndimage.median_filter(noisy, 3)

.. figure:: auto_examples/images/plot_lena_denoise_1.png
    :scale: 60
    :target: auto_examples/plot_lena_denoise.html

.. only:: html

    [:ref:`Python source code <example_plot_lena_denoise.py>`]


Median filter: better result for straight boundaries (**low curvature**)::

    >>> im = np.zeros((20, 20))
    >>> im[5:-5, 5:-5] = 1
    >>> im = ndimage.distance_transform_bf(im)
    >>> im_noise = im + 0.2*np.random.randn(*im.shape)
    >>> im_med = ndimage.median_filter(im_noise, 3)

.. figure:: auto_examples/images/plot_denoising_1.png
    :scale: 60
    :target: auto_examples/plot_denoising.html

.. only:: html

    [:ref:`Python source code <example_plot_denoising.py>`]


Other rank filter: ``ndimage.maximum_filter``,
``ndimage.percentile_filter``

Other local non-linear filters: Wiener (``scipy.signal.wiener``), etc.

**Non-local filters**

**Total-variation (TV) denoising**. Find a new image 
so that the total-variation of the image (integral of the norm L1 of
the gradient) is minimized, while being close to the measured image::

    >>> # from scikits.image.filter import tv_denoise
    >>> from tv_denoise import tv_denoise
    >>> tv_denoised = tv_denoise(noisy, weight=10)
    >>> # More denoising (to the expense of fidelity to data)
    >>> tv_denoised = tv_denoise(noisy, weight=50)

The total variation filter ``tv_denoise`` is available in the
``scikits.image``, (doc:
http://scikits-image.org/docs/dev/api/scikits.image.filter.html#tv-denoise),
but for convenience we've shipped it as a :download:`standalone module
<../../pyplots/tv_denoise.py>` with this tutorial.

.. figure:: auto_examples/images/plot_lena_tv_denoise_1.png
    :scale: 60
    :target: auto_examples/plot_lena_tv_denoise.html

.. only:: html

    [:ref:`Python source code <example_plot_lena_tv_denoise.py>`]


Mathematical morphology
-----------------------

See http://en.wikipedia.org/wiki/Mathematical_morphology

Probe an image with a simple shape (a **structuring element**), and
modify this image according to how the shape locally fits or misses the
image. 

**Structuring element**::

    >>> el = ndimage.generate_binary_structure(2, 1)
    >>> el
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> el.astype(np.int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

.. image:: diamond_kernel.png
    :align: center

**Erosion** = minimum filter. Replace the value of a pixel by the minimal value covered by the structuring element.::

    >>> a = np.zeros((7,7), dtype=np.int)
    >>> a[1:6, 2:5] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_erosion(a).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> #Erosion removes objects smaller than the structure
    >>> ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])


.. image:: morpho_mat.png
    :align: center


**Dilation**: maximum filter::

    >>> a = np.zeros((5, 5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])


Also works for grey-valued images::

    >>> np.random.seed(2)
    >>> x, y = (63*np.random.random((2, 8))).astype(np.int)
    >>> im[x, y] = np.arange(8)

    >>> bigger_points = ndimage.grey_dilation(im, size=(5, 5), structure=np.ones((5, 5)))

    >>> square = np.zeros((16, 16))
    >>> square[4:-4, 4:-4] = 1
    >>> dist = ndimage.distance_transform_bf(square)
    >>> dilate_dist = ndimage.grey_dilation(dist, size=(3, 3), \
    ...         structure=np.ones((3, 3)))


.. figure:: auto_examples/images/plot_greyscale_dilation_1.png
    :scale: 40
    :target: auto_examples/plot_greyscale_dilation.html

.. only:: html

    [:ref:`Python source code <example_plot_greyscale_dilation.py>`]

**Opening**: erosion + dilation::

    >>> a = np.zeros((5,5), dtype=np.int)
    >>> a[1:4, 1:4] = 1; a[4, 4] = 1
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> ndimage.binary_opening(a, structure=np.ones((3,3))).astype(np.int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening can also smooth corners
    >>> ndimage.binary_opening(a).astype(np.int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

**Application**: remove noise::

    >>> square = np.zeros((32, 32))
    >>> square[10:-10, 10:-10] = 1
    >>> np.random.seed(2)
    >>> x, y = (32*np.random.random((2, 20))).astype(np.int)
    >>> square[x, y] = 1

    >>> open_square = ndimage.binary_opening(square)

    >>> eroded_square = ndimage.binary_erosion(square)
    >>> reconstruction = ndimage.binary_propagation(eroded_square, mask=square)

.. figure:: auto_examples/images/plot_propagation_1.png
    :scale: 40
    :target: auto_examples/plot_propagation.html

.. only:: html

    [:ref:`Python source code <example_plot_propagation.py>`]

**Closing**: dilation + erosion

Many other mathematical morphology operations: hit and miss transform, tophat,
etc.

Feature extraction
==================

Edge detection
--------------

Synthetic data::

    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> 
    >>> im = ndimage.rotate(im, 15, mode='constant')
    >>> im = ndimage.gaussian_filter(im, 8)

Use a **gradient operator** (**Sobel**) to find high intensity variations::

    >>> sx = ndimage.sobel(im, axis=0, mode='constant')
    >>> sy = ndimage.sobel(im, axis=1, mode='constant')
    >>> sob = np.hypot(sx, sy)

.. figure:: auto_examples/images/plot_find_edges_1.png
    :scale: 40
    :target: auto_examples/plot_find_edges.html

.. only:: html

    [:ref:`Python source code <example_plot_find_edges.py>`]

**Canny filter**

The Canny filter is available in the ``scikits.image``
(`doc <http://scikits-image.org/docs/dev/api/scikits.image.filter.html#canny>`_),
but for convenience we've shipped it as a :download:`standalone module
<../../pyplots/image_source_canny.py>` with this tutorial. ::

  >>> #from scikits.image.filter import canny
  >>> #or use module shipped with tutorial
  >>> im += 0.1*np.random.random(im.shape)
  >>> edges = canny(im, 1, 0.4, 0.2) # not enough smoothing
  >>> edges = canny(im, 3, 0.3, 0.2) # better parameters

.. figure:: auto_examples/images/plot_canny_1.png
    :scale: 40
    :target: auto_examples/plot_canny.html

.. only:: html

    [:ref:`Python source code <example_plot_canny.py>`]

Several parameters need to be adjusted... risk of overfitting

Segmentation
------------

* **Histogram-based** segmentation (no spatial information)

::

    >>> n = 10
    >>> l = 256
    >>> im = np.zeros((l, l))
    >>> np.random.seed(1)
    >>> points = l*np.random.random((2, n**2))
    >>> im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    >>> im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

    >>> mask = (im > im.mean()).astype(np.float)
    >>> mask += 0.1 * im
    >>> img = mask + 0.2*np.random.randn(*mask.shape)

    >>> hist, bin_edges = np.histogram(img, bins=60)
    >>> bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    >>> binary_img = img > 0.5

.. figure:: auto_examples/images/plot_histo_segmentation_1.png
    :scale: 65
    :target: auto_examples/plot_histo_segmentation.html

.. only:: html

    [:ref:`Python source code <example_plot_histo_segmentation.py>`]

Automatic thresholding: use Gaussian mixture model::

    >>> mask = (im > im.mean()).astype(np.float)
    >>> mask += 0.1 * im
    >>> img = mask + 0.3*np.random.randn(*mask.shape)

    >>> from sklearn.mixture import GMM
    >>> classif = GMM(n_components=2)
    >>> classif.fit(img.reshape((img.size, 1))) # doctest: +ELLIPSIS
    GMM(...)

    >>> classif.means_
    array([[ 0.9353155 ],
           [-0.02966039]])
    >>> np.sqrt(classif.covars_).ravel()
    array([ 0.35074631,  0.28225327])
    >>> classif.weights_
    array([ 0.40989799,  0.59010201])
    >>> threshold = np.mean(classif.means_)
    >>> binary_img = img > threshold

.. image:: image_GMM.png
    :align: center
    :scale: 100

Use mathematical morphology to clean up the result::

    >>> # Remove small white regions
    >>> open_img = ndimage.binary_opening(binary_img)
    >>> # Remove small black hole
    >>> close_img = ndimage.binary_closing(open_img)

.. figure:: auto_examples/images/plot_clean_morpho_1.png
    :scale: 65
    :target: auto_examples/plot_clean_morpho.html

.. only:: html

    [:ref:`Python source code <example_plot_clean_morpho.py>`]

.. topic:: **Exercise**
    :class: green

    Check that reconstruction operations (erosion + propagation) produce a
    better result than opening/closing::

	>>> eroded_img = ndimage.binary_erosion(binary_img)
	>>> reconstruct_img = ndimage.binary_propagation(eroded_img, mask=binary_img)
	>>> tmp = np.logical_not(reconstruct_img)
	>>> eroded_tmp = ndimage.binary_erosion(tmp)
	>>> reconstruct_final = np.logical_not(ndimage.binary_propagation(eroded_tmp, mask=tmp))
	>>> np.abs(mask - close_img).mean()
	0.014678955078125
	>>> np.abs(mask - reconstruct_final).mean()
	0.0042572021484375

.. topic:: **Exercise**
    :class: green

    Check how a first denoising step (median filter, total variation)
    modifies the histogram, and check that the resulting histogram-based
    segmentation is more accurate.

* **Graph-based** segmentation: use spatial information.

::

    >>> from sklearn.feature_extraction import image
    >>> from sklearn.cluster import spectral_clustering

    >>> l = 100
    >>> x, y = np.indices((l, l))

    >>> center1 = (28, 24)
    >>> center2 = (40, 50)
    >>> center3 = (67, 58)
    >>> center4 = (24, 70)
    >>> radius1, radius2, radius3, radius4 = 16, 14, 15, 14

    >>> circle1 = (x - center1[0])**2 + (y - center1[1])**2 < radius1**2
    >>> circle2 = (x - center2[0])**2 + (y - center2[1])**2 < radius2**2
    >>> circle3 = (x - center3[0])**2 + (y - center3[1])**2 < radius3**2
    >>> circle4 = (x - center4[0])**2 + (y - center4[1])**2 < radius4**2

    >>> # 4 circles
    >>> img = circle1 + circle2 + circle3 + circle4
    >>> mask = img.astype(bool)
    >>> img = img.astype(float)

    >>> img += 1 + 0.2*np.random.randn(*img.shape)
    >>> # Convert the image into a graph with the value of the gradient on
    >>> # the edges.
    >>> graph = image.img_to_graph(img, mask=mask)

    >>> # Take a decreasing function of the gradient: we take it weakly
    >>> # dependant from the gradient the segmentation is close to a voronoi
    >>> graph.data = np.exp(-graph.data/graph.data.std())

    >>> labels = spectral_clustering(graph, k=4, mode='arpack')
    >>> label_im = -np.ones(mask.shape)
    >>> label_im[mask] = labels


.. image:: image_spectral_clustering.png
    :align: center



Measuring objects properties: ``ndimage.measurements``
========================================================

Synthetic data::

    >>> n = 10
    >>> l = 256
    >>> im = np.zeros((l, l))
    >>> points = l*np.random.random((2, n**2))
    >>> im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    >>> im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
    >>> mask = im > im.mean()

* **Analysis of connected components**

Label connected components: ``ndimage.label``:: 

    >>> label_im, nb_labels = ndimage.label(mask)
    >>> nb_labels # how many regions?
    23
    >>> plt.imshow(label_im)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>

.. figure:: auto_examples/images/plot_synthetic_data_1.png
    :scale: 90
    :target: auto_examples/plot_synthetic_data.html

.. only:: html

    [:ref:`Python source code <example_plot_synthetic_data.py>`]

Compute size, mean_value, etc. of each region::

    >>> sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    >>> mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))

Clean up small connect components::

    >>> mask_size = sizes < 1000
    >>> remove_pixel = mask_size[label_im]
    >>> remove_pixel.shape
    (256, 256)
    >>> label_im[remove_pixel] = 0
    >>> plt.imshow(label_im)        # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>

Now reassign labels with ``np.searchsorted``::

    >>> labels = np.unique(label_im)
    >>> label_im = np.searchsorted(labels, label_im)

.. figure:: auto_examples/images/plot_measure_data_1.png
    :scale: 90
    :target: auto_examples/plot_measure_data.html

.. only:: html

    [:ref:`Python source code <example_plot_measure_data.py>`]

Find region of interest enclosing object::

    >>> slice_x, slice_y = ndimage.find_objects(label_im==4)[0]
    >>> roi = im[slice_x, slice_y]
    >>> plt.imshow(roi)     # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>

.. figure:: auto_examples/images/plot_find_object_1.png
    :scale: 130
    :target: auto_examples/plot_find_object.html

.. only:: html

    [:ref:`Python source code <example_plot_find_object.py>`]

Other spatial measures: ``ndimage.center_of_mass``,
``ndimage.maximum_position``, etc.

Can be used outside the limited scope of segmentation applications. 

Example: block mean::

    >>> from scipy import misc
    >>> l = misc.lena()
    >>> sx, sy = l.shape
    >>> X, Y = np.ogrid[0:sx, 0:sy]
    >>> regions = sy/6 * (X/4) + Y/6  # note that we use broadcasting
    >>> block_mean = ndimage.mean(l, labels=regions, index=np.arange(1,
    ...     regions.max() +1))
    >>> block_mean.shape = (sx/4, sy/6)

.. figure:: auto_examples/images/plot_block_mean_1.png
    :scale: 70
    :target: auto_examples/plot_block_mean.html

.. only:: html

    [:ref:`Python source code <example_plot_block_mean.py>`]

When regions are regular blocks, it is more efficient to use stride
tricks (:ref:`stride-manipulation-label`).

Non-regularly-spaced blocks: radial mean::

    >>> sx, sy = l.shape
    >>> X, Y = np.ogrid[0:sx, 0:sy]
    >>> r = np.hypot(X - sx/2, Y - sy/2)
    >>> rbin = (20* r/r.max()).astype(np.int)
    >>> radial_mean = ndimage.mean(l, labels=rbin, index=np.arange(1, rbin.max() +1))

.. figure:: auto_examples/images/plot_radial_mean_1.png
    :scale: 70
    :target: auto_examples/plot_radial_mean.html

.. only:: html

    [:ref:`Python source code <example_plot_radial_mean.py>`]

* **Other measures** 

Correlation function, Fourier/wavelet spectrum, etc.

One example with mathematical morphology: **granulometry**
(http://en.wikipedia.org/wiki/Granulometry_%28morphology%29)

::

    >>> def disk_structure(n):
    ...     struct = np.zeros((2 * n + 1, 2 * n + 1))
    ...     x, y = np.indices((2 * n + 1, 2 * n + 1))
    ...     mask = (x - n)**2 + (y - n)**2 <= n**2
    ...     struct[mask] = 1
    ...     return struct.astype(np.bool)
    ... 
    >>> 
    >>> def granulometry(data, sizes=None):
    ...     s = max(data.shape)
    ...     if sizes == None:
    ...         sizes = range(1, s/2, 2)
    ...     granulo = [ndimage.binary_opening(data, \
    ...         structure=disk_structure(n)).sum() for n in sizes]
    ...     return granulo
    ... 
    >>> 
    >>> np.random.seed(1)
    >>> n = 10
    >>> l = 256
    >>> im = np.zeros((l, l))
    >>> points = l*np.random.random((2, n**2))
    >>> im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    >>> im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
    >>> 
    >>> mask = im > im.mean()
    >>> 
    >>> granulo = granulometry(mask, sizes=np.arange(2, 19, 4))

.. figure:: auto_examples/images/plot_granulo_1.png
    :scale: 100
    :target: auto_examples/plot_granulo.html

.. only:: html

    [:ref:`Python source code <example_plot_granulo.py>`]

