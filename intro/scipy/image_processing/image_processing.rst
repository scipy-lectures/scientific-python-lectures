:orphan:

.. for doctests
   >>> import matplotlib.pyplot as plt

:mod:`scipy.ndimage` provides manipulation of n-dimensional arrays as
images.

Geometrical transformations on images
.......................................

Changing orientation, resolution, .. ::

    >>> import scipy as sp

    >>> # Load an image
    >>> face = sp.datasets.face(gray=True)

    >>> # Shift, rotate and zoom it
    >>> shifted_face = sp.ndimage.shift(face, (50, 50))
    >>> shifted_face2 = sp.ndimage.shift(face, (50, 50), mode='nearest')
    >>> rotated_face = sp.ndimage.rotate(face, 30)
    >>> cropped_face = face[50:-50, 50:-50]
    >>> zoomed_face = sp.ndimage.zoom(face, 2)
    >>> zoomed_face.shape
    (1536, 2048)

.. image:: /intro/scipy/auto_examples/images/sphx_glr_plot_image_transform_001.png
    :target: auto_examples/plot_image_transform.html
    :scale: 70
    :align: center


::

    >>> plt.subplot(151)
    <Axes: >

    >>> plt.imshow(shifted_face, cmap=plt.cm.gray)
    <matplotlib.image.AxesImage object at 0x...>

    >>> plt.axis('off')
    (-0.5, 1023.5, 767.5, -0.5)

    >>> # etc.


Image filtering
...................

Generate a noisy face::

    >>> import scipy as sp
    >>> face = sp.datasets.face(gray=True)
    >>> face = face[:512, -512:]  # crop out square on right
    >>> import numpy as np
    >>> noisy_face = np.copy(face).astype(float)
    >>> rng = np.random.default_rng()
    >>> noisy_face += face.std() * 0.5 * rng.standard_normal(face.shape)

Apply a variety of filters on it::

    >>> blurred_face = sp.ndimage.gaussian_filter(noisy_face, sigma=3)
    >>> median_face = sp.ndimage.median_filter(noisy_face, size=5)
    >>> wiener_face = sp.signal.wiener(noisy_face, (5, 5))

.. image:: /intro/scipy/auto_examples/images/sphx_glr_plot_image_filters_001.png
    :target: auto_examples/plot_image_filters.html
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

.. image:: /intro/scipy/image_processing/morpho_mat.png
   :align: center

Mathematical-morphology operations use a *structuring element*
in order to modify geometrical structures.

Let us first generate a structuring element::

    >>> el = sp.ndimage.generate_binary_structure(2, 1)
    >>> el
    array([[False, True, False],
           [...True, True, True],
           [False, True, False]])
    >>> el.astype(int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

* **Erosion** :func:`scipy.ndimage.binary_erosion` ::

    >>> a = np.zeros((7, 7), dtype=int)
    >>> a[1:6, 2:5] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> sp.ndimage.binary_erosion(a).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # Erosion removes objects smaller than the structure
    >>> sp.ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype)
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
    array([[0.,  0.,  0.,  0.,  0.],
           [0.,  0.,  0.,  0.,  0.],
           [0.,  0.,  1.,  0.,  0.],
           [0.,  0.,  0.,  0.,  0.],
           [0.,  0.,  0.,  0.,  0.]])
    >>> sp.ndimage.binary_dilation(a).astype(a.dtype)
    array([[0.,  0.,  0.,  0.,  0.],
           [0.,  0.,  1.,  0.,  0.],
           [0.,  1.,  1.,  1.,  0.],
           [0.,  0.,  1.,  0.,  0.],
           [0.,  0.,  0.,  0.,  0.]])

* **Opening** :func:`scipy.ndimage.binary_opening` ::

    >>> a = np.zeros((5, 5), dtype=int)
    >>> a[1:4, 1:4] = 1
    >>> a[4, 4] = 1
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> sp.ndimage.binary_opening(a, structure=np.ones((3, 3))).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening can also smooth corners
    >>> sp.ndimage.binary_opening(a).astype(int)
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
    >>> rng = np.random.default_rng()
    >>> a += 0.25 * rng.standard_normal(a.shape)
    >>> mask = a>=0.5
    >>> opened_mask = sp.ndimage.binary_opening(mask)
    >>> closed_mask = sp.ndimage.binary_closing(opened_mask)

.. image:: /intro/scipy/auto_examples/images/sphx_glr_plot_mathematical_morpho_001.png
    :target: auto_examples/plot_mathematical_morpho.html
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

    >>> a = np.zeros((7, 7), dtype=int)
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
    >>> sp.ndimage.grey_erosion(a, size=(3, 3))
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

.. image:: /intro/scipy/auto_examples/images/sphx_glr_plot_connect_measurements_001.png
    :target: auto_examples/plot_connect_measurements.html
    :scale: 60
    :align: center

.. image:: /intro/scipy/auto_examples/images/sphx_glr_plot_connect_measurements_002.png
    :target: auto_examples/plot_connect_measurements.html
    :scale: 60
    :align: right

:func:`scipy.ndimage.label` assigns a different label to each connected
component::

    >>> labels, nb = sp.ndimage.label(mask)
    >>> nb
    8

.. raw:: html

   <div style="clear: both"></div>


Now compute measurements on each connected component::

    >>> areas = sp.ndimage.sum(mask, labels, range(1, labels.max()+1))
    >>> areas   # The number of pixels in each connected component
    array([190.,   45.,  424.,  278.,  459.,  190.,  549.,  424.])
    >>> maxima = sp.ndimage.maximum(sig, labels, range(1, labels.max()+1))
    >>> maxima  # The maximum signal in each connected component
    array([ 1.80238238,   1.13527605,   5.51954079,   2.49611818, 6.71673619,
            1.80238238,  16.76547217,   5.51954079])

.. image:: /intro/scipy/auto_examples/images/sphx_glr_plot_connect_measurements_003.png
    :target: auto_examples/plot_connect_measurements.html
    :scale: 60
    :align: right


Extract the 4th connected component, and crop the array around it::

    >>> sp.ndimage.find_objects(labels)[3]
    (slice(30, 48, None), slice(30, 48, None))
    >>> sl = sp.ndimage.find_objects(labels)[3]
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(sig[sl])
    <matplotlib.image.AxesImage object at ...>



See the summary exercise on :ref:`summary_exercise_image_processing` for a more
advanced example.
