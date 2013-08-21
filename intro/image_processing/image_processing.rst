The submodule dedicated to image processing in scipy is :mod:`scipy.ndimage`. ::

    >>> from scipy import ndimage

Image processing routines may be sorted according to the category of
processing they perform.


Geometrical transformations on images
.......................................

Changing orientation, resolution, .. ::

    >>> from scipy import misc
    >>> lena = misc.lena()
    >>> shifted_lena = ndimage.shift(lena, (50, 50))
    >>> shifted_lena2 = ndimage.shift(lena, (50, 50), mode='nearest')
    >>> rotated_lena = ndimage.rotate(lena, 30)
    >>> cropped_lena = lena[50:-50, 50:-50]
    >>> zoomed_lena = ndimage.zoom(lena, 2)
    >>> zoomed_lena.shape
    (1024, 1024)

.. figure:: image_processing/lena_transforms.png
   :align: center
   :scale: 70


.. sourcecode:: ipython

    In [35]: subplot(151)
    Out[35]: <matplotlib.axes.AxesSubplot object at 0x925f46c>

    In [36]: pl.imshow(shifted_lena, cmap=cm.gray)
    Out[36]: <matplotlib.image.AxesImage object at 0x9593f6c>

    In [37]: axis('off')
    Out[37]: (-0.5, 511.5, 511.5, -0.5)

    In [39]: # etc.


Image filtering
...................

::

    >>> from scipy import misc
    >>> lena = misc.lena()
    >>> import numpy as np
    >>> noisy_lena = np.copy(lena).astype(np.float)
    >>> noisy_lena += lena.std()*0.5*np.random.standard_normal(lena.shape)
    >>> blurred_lena = ndimage.gaussian_filter(noisy_lena, sigma=3)
    >>> median_lena = ndimage.median_filter(blurred_lena, size=5)
    >>> from scipy import signal
    >>> wiener_lena = signal.wiener(blurred_lena, (5,5))

.. figure:: image_processing/filtered_lena.png
   :align: center
   :scale: 80


Many other filters in :mod:`scipy.ndimage.filters` and :mod:`scipy.signal`
can be applied to images.

.. topic:: Exercise
   :class: green

    Compare histograms for the different filtered images.

Mathematical morphology
........................

Mathematical morphology is a mathematical theory that stems from set
theory. It characterizes and transforms geometrical structures. Binary
(black and white) images, in particular, can be transformed using this
theory: the sets to be transformed are the sets of neighboring
non-zero-valued pixels. The theory was also extended to gray-valued images.

.. image:: image_processing/morpho_mat.png
   :align: center

Elementary mathematical-morphology operations use a *structuring element*
in order to modify other geometrical structures.

Let us first generate a structuring element ::

    >>> el = ndimage.generate_binary_structure(2, 1)
    >>> el
    array([[False, True, False],
           [True, True, True],
           [False, True, False]], dtype=bool)
    >>> el.astype(np.int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

* **Erosion** ::

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

* **Dilation** ::

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

* **Opening** ::

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

* **Closing:** ``ndimage.binary_closing``

.. topic:: Exercise
   :class: green

    Check that opening amounts to eroding, then dilating.

An opening operation removes small structures, while a closing operation
fills small holes. Such operations can therefore be used to "clean" an
image. ::

    >>> a = np.zeros((50, 50))
    >>> a[10:-10, 10:-10] = 1
    >>> a += 0.25*np.random.standard_normal(a.shape)
    >>> mask = a>=0.5
    >>> opened_mask = ndimage.binary_opening(mask)
    >>> closed_mask = ndimage.binary_closing(opened_mask)

.. figure:: image_processing/morpho.png
   :align: center
   :scale: 75

.. topic:: Exercise
   :class: green

    Check that the area of the reconstructed square is smaller
    than the area of the initial square. (The opposite would occur if the
    closing step was performed *before* the opening).

For *gray-valued* images, eroding (resp. dilating) amounts to replacing
a pixel by the minimal (resp. maximal) value among pixels covered by the
structuring element centered on the pixel of interest. ::

    >>> a = np.zeros((7,7), dtype=np.int)
    >>> a[1:6, 1:6] = 3
    >>> a[4,4] = 2; a[2,3] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 1, 3, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 3, 2, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_erosion(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])


Measurements on images
........................

Let us first generate a nice synthetic binary image. ::

    >>> x, y = np.indices((100, 100))
    >>> sig = np.sin(2*np.pi*x/50.)*np.sin(2*np.pi*y/50.)*(1+x*y/50.**2)**2
    >>> mask = sig > 1

Now we look for various information about the objects in the image::

    >>> labels, nb = ndimage.label(mask)
    >>> nb
    8
    >>> areas = ndimage.sum(mask, labels, xrange(1, labels.max()+1))
    >>> areas
    array([ 190.,   45.,  424.,  278.,  459.,  190.,  549.,  424.])
    >>> maxima = ndimage.maximum(sig, labels, xrange(1, labels.max()+1))
    >>> maxima
    array([  1.80238238,   1.13527605,   5.51954079,   2.49611818,
             6.71673619,   1.80238238,  16.76547217,   5.51954079])
    >>> ndimage.find_objects(labels==4)
    [(slice(30L, 48L, None), slice(30L, 48L, None))]
    >>> sl = ndimage.find_objects(labels==4)
    >>> import pylab as pl
    >>> pl.imshow(sig[sl[0]])   # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>


.. figure:: image_processing/measures.png
   :align: center
   :scale: 80


See the summary exercise on :ref:`summary_exercise_image_processing` for a more
advanced example.


