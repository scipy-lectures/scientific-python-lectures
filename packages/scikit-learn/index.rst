.. _scikit-learn_chapter:

========================================
scikit-learn: machine learning in Python
========================================

**Authors**: *Gael Varoquaux*

.. image:: images/scikit-learn-logo.png
   :scale: 40
   :align: right

.. topic:: Prerequisites

    * Numpy, Scipy
    * IPython
    * matplotlib
    * scikit-learn (http://scikit-learn.org)

.. seealso:: **Statistics in Python**

    The content of the :ref:`statistics` chapter may also be of interest
    for readers looking into machine learning.

.. contents:: Chapters contents
   :local:
   :depth: 2

.. For doctests
   >>> import numpy as np
   >>> np.random.seed(0)
   >>> # For doctest on headless environments
   >>> from matplotlib import pyplot as plt
   >>> plt.switch_backend('Agg')

Introduction: problem settings
===============================

What is machine learning?
--------------------------

.. tip::

    Machine Learning is about building programs with **tunable
    parameters** that are adjusted automatically so as to improve their
    behavior by **adapting to previously seen data.**

    Machine Learning can be considered a subfield of **Artificial
    Intelligence** since those algorithms can be seen as building blocks
    to make computers learn to behave more intelligently by somehow
    **generalizing** rather that just storing and retrieving data items
    like a database system would do.

.. figure:: auto_examples/images/sphx_glr_plot_separator_001.png
   :align: right
   :target: auto_examples/plot_separator.html
   :width: 350

   A classification problem

We'll take a look at two very simple machine learning tasks here. The
first is a **classification** task: the figure shows a collection of
two-dimensional data, colored according to two different class labels. A
classification algorithm may be used to draw a dividing boundary between
the two clusters of points:

By drawing this separating line, we have learned a model which can
**generalize** to new data: if you were to drop another point onto the
plane which is unlabeled, this algorithm could now **predict** whether
it's a blue or a red point.

.. raw:: html

   <div style="flush: both;"></div>

.. figure:: auto_examples/images/sphx_glr_plot_linear_regression_001.png
   :align: right
   :target: auto_examples/plot_linear_regression.html
   :width: 350

   A regression problem

|

The next simple task we'll look at is a **regression** task: a simple
best-fit line to a set of data.

Again, this is an example of fitting a model to data, but our focus here
is that the model can make generalizations about new data. The model has
been **learned** from the training data, and can be used to predict the
result of test data: here, we might be given an x-value, and the model
would allow us to predict the y value.

Data in scikit-learn
---------------------

The data matrix
~~~~~~~~~~~~~~~~

Machine learning algorithms implemented in scikit-learn expect data
to be stored in a **two-dimensional array or matrix**. The arrays can be
either ``numpy`` arrays, or in some cases ``scipy.sparse`` matrices. The
size of the array is expected to be ``[n_samples, n_features]``

-  **n\_samples:** The number of samples: each sample is an item to
   process (e.g. classify). A sample can be a document, a picture, a
   sound, a video, an astronomical object, a row in database or CSV
   file, or whatever you can describe with a fixed set of quantitative
   traits.
-  **n\_features:** The number of features or distinct traits that can
   be used to describe each item in a quantitative manner. Features are
   generally real-valued, but may be boolean or discrete-valued in some
   cases.

.. tip::

    The number of features must be fixed in advance. However it can be
    very high dimensional (e.g. millions of features) with most of them
    being zeros for a given sample. This is a case where ``scipy.sparse``
    matrices can be useful, in that they are much more memory-efficient
    than numpy arrays.

A Simple Example: the Iris Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The application problem
........................

As an example of a simple dataset, we are going to take a look at the
iris data stored by scikit-learn. The data consists of measurements of
three different species of irises. There are three species of iris in
the dataset:

.. |setosa_picture| image:: images/iris_setosa.jpg
    
.. |versicolor_picture| image:: images/iris_versicolor.jpg
    
.. |virginica_picture| image:: images/iris_virginica.jpg

===================== ===================== =====================
|setosa_picture|      |versicolor_picture|  |virginica_picture|
===================== ===================== =====================
Setosa Iris           Versicolor Iris       Virginica Iris
===================== ===================== =====================


.. topic:: **Quick Question:**

    **If we want to design an algorithm to recognize iris species, what
    might the data be?**

    Remember: we need a 2D array of size ``[n_samples x n_features]``.

    -  What would the ``n_samples`` refer to?

    -  What might the ``n_features`` refer to?

Remember that there must be a **fixed** number of features for each
sample, and feature number ``i`` must be a similar kind of quantity for
each sample.

Loading the Iris Data with Scikit-learn
........................................

Scikit-learn has a very straightforward set of data on these iris
species. The data consist of the following:

-  Features in the Iris dataset:

    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm

-  Target classes to predict:

    1. Iris Setosa
    2. Iris Versicolour
    3. Iris Virginica

``scikit-learn`` embeds a copy of the iris CSV file along with a helper
function to load it into numpy arrays::

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()

The features of each sample flower are stored in the ``data`` attribute
of the dataset::

    >>> print(iris.data.shape)
    (150, 4)
    >>> n_samples, n_features = iris.data.shape
    >>> print(n_samples)
    150
    >>> print(n_features)
    4
    >>> print(iris.data[0])
    [ 5.1  3.5  1.4  0.2]

The information about the class of each sample is stored in the
``target`` attribute of the dataset::

    >>> print(iris.target.shape)
    (150,)
    >>> print(iris.target)
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]

The names of the classes are stored in the last attribute, namely
``target_names``::

    >>> print(iris.target_names)
    ['setosa' 'versicolor' 'virginica']

This data is four-dimensional, but we can visualize two of the
dimensions at a time using a simple scatter-plot: 

.. image:: auto_examples/images/sphx_glr_plot_iris_scatter_001.png
   :align: center
   :target: auto_examples/plot_iris_scatter.html

.. topic:: **Excercise**:
    :class: green
   
    Can you choose 2 features to find a plot where it is easier to
    seperate the different classes of irises?

    **Hint**: click on the figure above to see the code that generates it,
    and modify this code.


