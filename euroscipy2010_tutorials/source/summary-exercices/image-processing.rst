.. _summary_exercise_image_processing:

Image processing application: counting bubbles and unmolten grains
------------------------------------------------------------------

.. image:: ../image_processing/MV_HFV_012.jpg
   :align: center
   :scale: 70

.. only:: latex

Statement of the problem
..........................

1. Open the image file MV_HFV_012.jpg and display it. Browse through the
keyword arguments in the docstring of ``imshow`` to display the image
with the "right" orientation (origin in the bottom left corner, and not
the upper left corner as for standard arrays).

This Scanning Element Microscopy image shows a glass sample (light gray
matrix) with some bubbles (on black) and unmolten sand grains (dark
gray). We wish to determine the fraction of the sample covered by these
three phases, and to estimate the typical size of sand grains and
bubbles, their sizes, etc.

2. Crop the image to remove the lower panel with measure information.

3. Slightly filter the image with a median filter in order to refine its
histogram. Check how the histogram changes.

4. Using the histogram of the filtered image, determine thresholds that
allow to define masks for sand pixels, glass pixels and bubble pixels.
Other option (homework): write a function that determines automatically
the thresholds from the minima of the histogram.

5. Display an image in which the three phases are colored with three
different colors.

6. Use mathematical morphology to clean the different phases.

7. Attribute labels to all bubbles and sand grains, and remove from the
sand mask grains that are smaller than 10 pixels. To do so, use
``ndimage.sum`` or ``np.bincount`` to compute the grain sizes.

8. Compute the mean size of bubbles.

.. only:: latex

   .. _image-answers:

Proposed solution
....................

