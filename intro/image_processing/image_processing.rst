:orphan:

.. for doctests
   >>> import matplotlib.pyplot as plt
   >>> plt.switch_backend("Agg")

:mod:`scipy.ndimage` provides manipulation of n-dimensional arrays as
images.

Geometrical transformations on images
.......................................

Changing orientation, resolution, .. ::

    >>> from scipy import misc  # Load an image
    >>> face = misc.face(gray=True)

    >>> from scipy import ndimage # Shift, roate and zoom it
    >>> shifted_face = ndimage.shift(face, (50, 50))
    >>> shifted_face2 = ndimage.shift(face, (50, 50), mode='nearest')
    >>> rotated_face = ndimage.rotate(face, 30)
    >>> cropped_face = face[50:-50, 50:-50]
    >>> zoomed_face = ndimage.zoom(face, 2)
    >>> zoomed_face.shape
    (1536, 2048)

.. image:: scipy/auto_examples/images/sphx_glr_plot_image_transform_001.png
    :target: scipy/auto_examples/plot_image_transform.html
    :scale: 70
    :align: center


::

    >>> plt.subplot(151)    # doctest: +ELLIPSIS
    <matplotlib.axes._subplots.AxesSubplot object at 0x...>

    >>> plt.imshow(shifted_face, cmap=plt.cm.gray)    # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at 0x...>

    >>> plt.axis('off')
    (-0.5, 1023.5, 767.5, -0.5)

    >>> # etc.


Image filtering
...................

Generate a noisy face::

    >>> from scipy import misc
    >>> face = misc.face(gray=True)
    >>> face = face[:512, -512:]  # crop out square on right
    >>> import numpy as np
    >>> noisy_face = np.copy(face).astype(np.float)
    >>> noisy_face += face.std() * 0.5 * np.random.standard_normal(face.shape)

Apply a variety of filters on it::

    >>> blurred_face = ndimage.gaussian_filter(noisy_face, sigma=3)
    >>> median_face = ndimage.median_filter(noisy_face, size=5)
    >>> from scipy import signal
    >>> wiener_face = signal.wiener(noisy_face, (5, 5))

.. image:: scipy/auto_examples/images/sphx_glr_plot_image_filters_001.png
    :target: scipy/auto_examples/plot_image_filters.html
    :scale: 70
    :align: center


Other filters in :mod:`scipy.ndimage.filters` and :mod:`scipy.signal`
can be applied to images.

.. topic:: Exercise
   :class: green

    Compare histograms for the different filtered images.

Mathematical morphology
........................

.. tip::

    `Mathematical morphology
    <https://en.wikipedia.org/wiki/Mathematical_morphology>`_ stems from set
    theory. It characterizes and transforms geometrical structures. Binary
    (black and white) images, in particular, can be transformed using this
    theory: the sets to be transformed are the sets of neighboring
    non-zero-valued pixels. The theory was also extended to gray-valued
    images.

.. image:: image_processing/morpho_mat.png
   :align: center

Mathematical-morphology operations use a *structuring element*
in order to modify geometrical structures.

Let us first generate a structuring element::

    >>> el = ndimage.generate_binary_structure(2, 1)
    >>> el # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[False, True, False],
           [...True, True, True],
           [False, True, False]], dtype=bool)
    >>> el.astype(np.int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

* **Erosion** :func:`scipy.ndimage.binary_erosion` ::

    >>> a = np.zeros((7, 7), dtype=np.int)
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

* **Dilation** :func:`scipy.ndimage.binary_dilation`  ::

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

* **Opening** :func:`scipy.ndimage.binary_opening` ::

    >>> a = np.zeros((5, 5), dtype=np.int)
    >>> a[1:4, 1:4] = 1
    >>> a[4, 4] = 1
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> ndimage.binary_opening(a, structure=np.ones((3, 3))).astype(np.int)
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

* **Closing:** :func:`scipy.ndimage.binary_closing`

.. topic:: Exercise
   :class: green

    Check that opening amounts to eroding, then dilating.

An opening operation removes small structures, while a closing operation
fills small holes. Such operations can therefore be used to "clean" an
image. ::

    >>> a = np.zeros((50, 50))
    >>> a[10:-10, 10:-10] = 1
    >>> a += 0.25 * np.random.standard_normal(a.shape)
    >>> mask = a>=0.5
    >>> opened_mask = ndimage.binary_opening(mask)
    >>> closed_mask = ndimage.binary_closing(opened_mask)

.. image:: scipy/auto_examples/images/sphx_glr_plot_mathematical_morpho_001.png
    :target: scipy/auto_examples/plot_mathematical_morpho.html
    :scale: 70
    :align: center


.. topic:: Exercise
   :class: green

    Check that the area of the reconstructed square is smaller
    than the area of the initial square. (The opposite would occur if the
    closing step was performed *before* the opening).

For *gray-valued* images, eroding (resp. dilating) amounts to replacing
a pixel by the minimal (resp. maximal) value among pixels covered by the
structuring element centered on the pixel of interest. ::

    >>> a = np.zeros((7, 7), dtype=np.int)
    >>> a[1:6, 1:6] = 3
    >>> a[4, 4] = 2; a[2, 3] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 1, 3, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 3, 2, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_erosion(a, size=(3, 3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])


Connected components and measurements on images
................................................

Let us first generate a nice synthetic binary image. ::

    >>> x, y = np.indices((100, 100))
    >>> sig = np.sin(2*np.pi*x/50.) * np.sin(2*np.pi*y/50.) * (1+x*y/50.**2)**2
    >>> mask = sig > 1

.. image:: scipy/auto_examples/images/sphx_glr_plot_connect_measurements_001.png
    :target: scipy/auto_examples/plot_connect_measurements.html
    :scale: 60
    :align: center

.. image:: scipy/auto_examples/images/sphx_glr_plot_connect_measurements_002.png
    :target: scipy/auto_examples/plot_connect_measurements.html
    :scale: 60
    :align: right

:func:`scipy.ndimage.label` assigns a different label to each connected
component::

    >>> labels, nb = ndimage.label(mask)
    >>> nb
    8

.. raw:: html

   <div style="clear: both"></div>


Now compute measurements on each connected component::

    >>> areas = ndimage.sum(mask, labels, range(1, labels.max()+1))
    >>> areas   # The number of pixels in each connected component
    array([ 190.,   45.,  424.,  278.,  459.,  190.,  549.,  424.])
    >>> maxima = ndimage.maximum(sig, labels, range(1, labels.max()+1))
    >>> maxima  # The maximum signal in each connected component
    array([  1.80238238,   1.13527605,   5.51954079,   2.49611818,
             6.71673619,   1.80238238,  16.76547217,   5.51954079])

.. image:: scipy/auto_examples/images/sphx_glr_plot_connect_measurements_003.png
    :target: scipy/auto_examples/plot_connect_measurements.html
    :scale: 60
    :align: right


Extract the 4th connected component, and crop the array around it::

    >>> ndimage.find_objects(labels==4) # doctest: +SKIP
    [(slice(30L, 48L, None), slice(30L, 48L, None))]
    >>> sl = ndimage.find_objects(labels==4)
    >>> from matplotlib import pyplot as plt
    >>> plt.imshow(sig[sl[0]])   # doctest: +ELLIPSIS
    <matplotlib.image.AxesImage object at ...>



See the summary exercise on :ref:`summary_exercise_image_processing` for a more
advanced example.


