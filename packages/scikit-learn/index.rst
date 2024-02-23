.. _scikit-learn_chapter:

========================================
scikit-learn: machine learning in Python
========================================

**Authors**: *Gael Varoquaux*

.. image:: images/scikit-learn-logo.png
   :scale: 40
   :align: right

.. topic:: Prerequisites

   .. rst-class:: horizontal

    * :ref:`numpy <numpy>`
    * :ref:`scipy <scipy>`
    * :ref:`matplotlib (optional) <matplotlib>`
    * :ref:`ipython (the enhancements come handy) <interactive_work>`

.. sidebar:: **Acknowledgements**

   This chapter is adapted from `a tutorial
   <https://www.youtube.com/watch?v=r4bRUvvlaBw>`__ given by Gaël
   Varoquaux, Jake Vanderplas, Olivier Grisel.

.. seealso:: **Data science in Python**

  * The :ref:`statistics` chapter may also be of interest
    for readers looking into machine learning.

  * The `documentation of scikit-learn <https://scikit-learn.org>`_ is
    very complete and didactic.

.. contents:: Chapters contents
   :local:
   :depth: 1

.. For doctests
   >>> import numpy as np
   >>> # For doctest on headless environments
   >>> import matplotlib.pyplot as plt

.. currentmodule:: sklearn

Introduction: problem settings
==============================

What is machine learning?
-------------------------

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
--------------------

The data matrix
~~~~~~~~~~~~~~~

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
    than NumPy arrays.

A Simple Example: the Iris Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The application problem
.......................

As an example of a simple dataset, let us a look at the
iris data stored by scikit-learn. Suppose we want to recognize species of
irises. The data consists of measurements of
three different species of irises:

.. |setosa_picture| image:: images/iris_setosa.jpg

.. |versicolor_picture| image:: images/iris_versicolor.jpg

.. |virginica_picture| image:: images/iris_virginica.jpg

===================== ===================== =====================
|setosa_picture|      |versicolor_picture|  |virginica_picture|
===================== ===================== =====================
Setosa Iris           Versicolor Iris       Virginica Iris
===================== ===================== =====================


.. topic:: **Quick Question:**
   :class: green

    **If we want to design an algorithm to recognize iris species, what
    might the data be?**

    Remember: we need a 2D array of size ``[n_samples x n_features]``.

    -  What would the ``n_samples`` refer to?

    -  What might the ``n_features`` refer to?

Remember that there must be a **fixed** number of features for each
sample, and feature number ``i`` must be a similar kind of quantity for
each sample.

Loading the Iris Data with Scikit-learn
.......................................

Scikit-learn has a very straightforward set of data on these iris
species. The data consist of the following:

-  Features in the Iris dataset:

   .. rst-class:: horizontal

    * sepal length (cm)
    * sepal width (cm)
    * petal length (cm)
    * petal width (cm)

-  Target classes to predict:

   .. rst-class:: horizontal

    * Setosa
    * Versicolour
    * Virginica

:mod:`scikit-learn` embeds a copy of the iris CSV file along with a
function to load it into NumPy arrays::

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()

.. note::

   **Import sklearn** Note that scikit-learn is imported as :mod:`sklearn`

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
    [5.1  3.5  1.4  0.2]

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
dimensions at a time using a scatter plot:

.. image:: auto_examples/images/sphx_glr_plot_iris_scatter_001.png
   :align: left
   :target: auto_examples/plot_iris_scatter.html

.. topic:: **Exercise**:
    :class: green

    Can you choose 2 features to find a plot where it is easier to
    separate the different classes of irises?

    **Hint**: click on the figure above to see the code that generates it,
    and modify this code.


Basic principles of machine learning with scikit-learn
======================================================

Introducing the scikit-learn estimator object
----------------------------------------------

Every algorithm is exposed in scikit-learn via an ''Estimator'' object.
For instance a linear regression is: :class:`sklearn.linear_model.LinearRegression` ::

    >>> from sklearn.linear_model import LinearRegression

**Estimator parameters**: All the parameters of an estimator can be set
when it is instantiated::

    >>> model = LinearRegression(n_jobs=1)
    >>> print(model)
    LinearRegression(n_jobs=1)

Fitting on data
~~~~~~~~~~~~~~~

Let's create some simple data with :ref:`numpy <numpy>`::

    >>> import numpy as np
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2])

    >>> X = x[:, np.newaxis] # The input data for sklearn is 2D: (samples == 3 x features == 1)
    >>> X
    array([[0],
           [1],
           [2]])

    >>> model.fit(X, y)
    LinearRegression(n_jobs=1)

**Estimated parameters**: When data is fitted with an estimator,
parameters are estimated from the data at hand. All the estimated
parameters are attributes of the estimator object ending by an
underscore::

    >>> model.coef_
    array([1.])

Supervised Learning: Classification and regression
--------------------------------------------------

In **Supervised Learning**, we have a dataset consisting of both
features and labels. The task is to construct an estimator which is able
to predict the label of an object given the set of features. A
relatively simple example is predicting the species of iris given a set
of measurements of its flower. This is a relatively simple task. Some
more complicated examples are:

-  given a multicolor image of an object through a telescope, determine
   whether that object is a star, a quasar, or a galaxy.
-  given a photograph of a person, identify the person in the photo.
-  given a list of movies a person has watched and their personal rating
   of the movie, recommend a list of movies they would like (So-called
   *recommender systems*: a famous example is the `Netflix
   Prize <https://en.wikipedia.org/wiki/Netflix_prize>`__).

.. tip::

    What these tasks have in common is that there is one or more unknown
    quantities associated with the object which needs to be determined from
    other observed quantities.

Supervised learning is further broken down into two categories,
**classification** and **regression**. In classification, the label is
discrete, while in regression, the label is continuous. For example, in
astronomy, the task of determining whether an object is a star, a
galaxy, or a quasar is a classification problem: the label is from three
distinct categories. On the other hand, we might wish to estimate the
age of an object based on such observations: this would be a regression
problem, because the label (age) is a continuous quantity.

**Classification**: K nearest neighbors (kNN) is one of the simplest
learning strategies: given a new, unknown observation, look up in your
reference database which ones have the closest features and assign the
predominant class. Let's try it out on our iris classification problem::

    from sklearn import neighbors, datasets
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    # What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
    print(iris.target_names[knn.predict([[3, 5, 4, 2]])])


.. figure:: auto_examples/images/sphx_glr_plot_iris_knn_001.png
   :align: center
   :target: auto_examples/plot_iris_knn.html

   A plot of the sepal space and the prediction of the KNN

**Regression**: The simplest possible regression setting is the linear
regression one:

.. literalinclude:: examples/plot_linear_regression.py
    :start-after: import matplotlib.pyplot as plt
    :end-before: plot the results

.. figure:: auto_examples/images/sphx_glr_plot_linear_regression_001.png
   :align: center
   :target: auto_examples/plot_linear_regression.html

   A plot of a simple linear regression.

A recap on Scikit-learn's estimator interface
---------------------------------------------

Scikit-learn strives to have a uniform interface across all methods, and
we’ll see examples of these below. Given a scikit-learn *estimator*
object named ``model``, the following methods are available:

:In **all Estimators**:

  - ``model.fit()`` : fit training data. For supervised learning
    applications, this accepts two arguments: the data ``X`` and the
    labels ``y`` (e.g. ``model.fit(X, y)``). For unsupervised learning
    applications, this accepts only a single argument, the data ``X``
    (e.g. ``model.fit(X)``).

:In **supervised estimators**:

  - ``model.predict()`` : given a trained model, predict the label of a
    new set of data. This method accepts one argument, the new data
    ``X_new`` (e.g. ``model.predict(X_new)``), and returns the learned
    label for each object in the array.
  - ``model.predict_proba()`` : For classification problems, some
    estimators also provide this method, which returns the probability
    that a new observation has each categorical label. In this case, the
    label with the highest probability is returned by
    ``model.predict()``.
  - ``model.score()`` : for classification or regression problems, most
    (all?) estimators implement a score method. Scores are between 0 and
    1, with a larger score indicating a better fit.

:In **unsupervised estimators**:

  - ``model.transform()`` : given an unsupervised model, transform new
    data into the new basis. This also accepts one argument ``X_new``,
    and returns the new representation of the data based on the
    unsupervised model.
  - ``model.fit_transform()`` : some estimators implement this method,
    which more efficiently performs a fit and a transform on the same
    input data.

Regularization: what it is and why it is necessary
--------------------------------------------------

Preferring simpler models
~~~~~~~~~~~~~~~~~~~~~~~~~

**Train errors** Suppose you are using a 1-nearest neighbor estimator.
How many errors do you expect on your train set?

-  Train set error is not a good measurement of prediction performance.
   You need to leave out a test set.
-  In general, we should accept errors on the train set.

**An example of regularization** The core idea behind regularization is
that we are going to prefer models that are simpler, for a certain
definition of ''simpler'', even if they lead to more errors on the train
set.

As an example, let's generate with a 9th order polynomial, with noise:

.. figure:: auto_examples/images/sphx_glr_plot_polynomial_regression_001.png
   :align: center
   :scale: 90
   :target: auto_examples/plot_polynomial_regression.html

And now, let's fit a 4th order and a 9th order polynomial to the data.

.. figure:: auto_examples/images/sphx_glr_plot_polynomial_regression_002.png
   :align: center
   :scale: 90
   :target: auto_examples/plot_polynomial_regression.html

With your naked eyes, which model do you prefer, the 4th order one, or
the 9th order one?

Let's look at the ground truth:

.. figure:: auto_examples/images/sphx_glr_plot_polynomial_regression_003.png
   :align: center
   :scale: 90
   :target: auto_examples/plot_polynomial_regression.html

.. tip::

    Regularization is ubiquitous in machine learning. Most scikit-learn
    estimators have a parameter to tune the amount of regularization. For
    instance, with k-NN, it is 'k', the number of nearest neighbors used to
    make the decision. k=1 amounts to no regularization: 0 error on the
    training set, whereas large k will push toward smoother decision
    boundaries in the feature space.

Simple versus complex models for classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |linear| image:: auto_examples/images/sphx_glr_plot_svm_non_linear_001.png
   :width: 400
   :target: auto_examples/plot_svm_non_linear.html

.. |nonlinear| image:: auto_examples/images/sphx_glr_plot_svm_non_linear_002.png
   :width: 400
   :target: auto_examples/plot_svm_non_linear.html

========================== ==========================
|linear|                   |nonlinear|
========================== ==========================
A linear separation        A non-linear separation
========================== ==========================

.. tip::

   For classification models, the decision boundary, that separates the
   class expresses the complexity of the model. For instance, a linear
   model, that makes a decision based on a linear combination of
   features, is more complex than a non-linear one.


Supervised Learning: Classification of Handwritten Digits
=========================================================

The nature of the data
-----------------------

.. sidebar:: Code and notebook

   Python code and Jupyter notebook for this section are found
   :ref:`here <sphx_glr_packages_scikit-learn_auto_examples_plot_digits_simple_classif.py>`


In this section we'll apply scikit-learn to the classification of
handwritten digits. This will go a bit beyond the iris classification we
saw before: we'll discuss some of the metrics which can be used in
evaluating the effectiveness of a classification model. ::

    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits()

.. image:: auto_examples/images/sphx_glr_plot_digits_simple_classif_001.png
   :target: auto_examples/plot_digits_simple_classif.html
   :align: center

Let us visualize the data and remind us what we're looking at (click on
the figure for the full code)::

    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

Visualizing the Data on its principal components
-------------------------------------------------

A good first-step for many problems is to visualize the data using a
*Dimensionality Reduction* technique. We'll start with the most
straightforward one, `Principal Component Analysis (PCA)
<https://en.wikipedia.org/wiki/Principal_component_analysis>`_.

PCA seeks orthogonal linear combinations of the features which show the
greatest variance, and as such, can help give you a good idea of the
structure of the data set. ::

    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=2)
    >>> proj = pca.fit_transform(digits.data)
    >>> plt.scatter(proj[:, 0], proj[:, 1], c=digits.target)
    <matplotlib.collections.PathCollection object at ...>
    >>> plt.colorbar()
    <matplotlib.colorbar.Colorbar object at ...>

.. image:: auto_examples/images/sphx_glr_plot_digits_simple_classif_002.png
   :align: center
   :target: auto_examples/plot_digits_simple_classif.html

.. topic:: **Question**
    :class: green

    Given these projections of the data, which numbers do you think a
    classifier might have trouble distinguishing?

Gaussian Naive Bayes Classification
-----------------------------------

For most classification problems, it's nice to have a simple, fast
method to provide a quick baseline classification. If the simple
and fast method is sufficient, then we don't have to waste CPU cycles on
more complex models. If not, we can use the results of the simple method
to give us clues about our data.

One good method to keep in mind is Gaussian Naive Bayes
(:class:`sklearn.naive_bayes.GaussianNB`).

.. sidebar:: Old scikit-learn versions

   :func:`~sklearn.model_selection.train_test_split` is imported from
   ``sklearn.cross_validation``

.. tip::

   Gaussian Naive Bayes fits a Gaussian distribution to each training label
   independently on each feature, and uses this to quickly give a rough
   classification. It is generally not sufficiently accurate for real-world
   data, but can perform surprisingly well, for instance on text data.

::

    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.model_selection import train_test_split

    >>> # split the data into training and validation sets
    >>> X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

    >>> # train the model
    >>> clf = GaussianNB()
    >>> clf.fit(X_train, y_train)
    GaussianNB()

    >>> # use the model to predict the labels of the test data
    >>> predicted = clf.predict(X_test)
    >>> expected = y_test
    >>> print(predicted)
    [5 1 7 2 8 9 4 3 9 3 6 2 3 2 6 7 4 3 5 7 5 7 0 1 2 5 9 8 1 8...]
    >>> print(expected)
    [5 8 7 2 8 9 4 3 7 3 6 2 3 2 6 7 4 3 5 7 5 7 0 1 2 5 3 3 1 8...]

As above, we plot the digits with the predicted labels to get an idea of
how well the classification is working.

.. image:: auto_examples/images/sphx_glr_plot_digits_simple_classif_003.png
   :align: center
   :target: auto_examples/plot_digits_simple_classif.html


.. topic:: **Question**
    :class: green

    Why did we split the data into training and validation sets?

Quantitative Measurement of Performance
---------------------------------------

We'd like to measure the performance of our estimator without having to
resort to plotting examples. A simple method might be to simply compare
the number of matches::

    >>> matches = (predicted == expected)
    >>> print(matches.sum())
    371
    >>> print(len(matches))
    450
    >>> matches.sum() / float(len(matches))
    0.82444...

We see that more than 80% of the 450 predictions match the input. But
there are other more sophisticated metrics that can be used to judge the
performance of a classifier: several are available in the
:mod:`sklearn.metrics` submodule.

One of the most useful metrics is the ``classification_report``, which
combines several measures and prints a table with the results::

    >>> from sklearn import metrics
    >>> print(metrics.classification_report(expected, predicted))
                precision    recall  f1-score   support
    <BLANKLINE>
           0       1.00      0.98      0.99        45
           1       0.91      0.66      0.76        44
           2       0.91      0.56      0.69        36
           3       0.89      0.67      0.77        49
           4       0.95      0.83      0.88        46
           5       0.93      0.93      0.93        45
           6       0.92      0.98      0.95        47
           7       0.75      0.96      0.84        50
           8       0.49      0.97      0.66        39
           9       0.85      0.67      0.75        49
    <BLANKLINE>
        accuracy                           0.82       450
       macro avg       0.86      0.82      0.82       450
    weighted avg       0.86      0.82      0.83       450
    <BLANKLINE>


Another enlightening metric for this sort of multi-label classification
is a *confusion matrix*: it helps us visualize which labels are being
interchanged in the classification errors::

    >>> print(metrics.confusion_matrix(expected, predicted))
    [[44  0  0  0  0  0  0  0  0  1]
     [ 0 29  0  0  0  0  1  6  6  2]
     [ 0  1 20  1  0  0  0  0 14  0]
     [ 0  0  0 33  0  2  0  1 11  2]
     [ 0  0  0  0 38  1  2  4  1  0]
     [ 0  0  0  0  0 42  1  0  2  0]
     [ 0  0  0  0  0  0 46  0  1  0]
     [ 0  0  0  0  1  0  0 48  0  1]
     [ 0  1  0  0  0  0  0  0 38  0]
     [ 0  1  2  3  1  0  0  5  4 33]]

We see here that in particular, the numbers 1, 2, 3, and 9 are often
being labeled 8.


Supervised Learning: Regression of Housing Data
===============================================

Here we'll do a short example of a regression problem: learning a
continuous value from a set of features.

A quick look at the data
-------------------------

.. sidebar:: Code and notebook

   Python code and Jupyter notebook for this section are found
   :ref:`here <sphx_glr_packages_scikit-learn_auto_examples_plot_california_prediction.py>`



We'll use the California house prices set, available in scikit-learn.
This records measurements of 8 attributes of housing markets in
California, as well as the median price. The question is: can you predict
the price of a new market given its attributes?::

    >>> from sklearn.datasets import fetch_california_housing
    >>> data = fetch_california_housing(as_frame=True)
    >>> print(data.data.shape)
    (20640, 8)
    >>> print(data.target.shape)
    (20640,)

We can see that there are just over 20000 data points.

The ``DESCR`` variable has a long description of the dataset::

    >>> print(data.DESCR)
    .. _california_housing_dataset:
    <BLANKLINE>
    California Housing dataset
    --------------------------
    <BLANKLINE>
    **Data Set Characteristics:**
    <BLANKLINE>
        :Number of Instances: 20640
    <BLANKLINE>
        :Number of Attributes: 8 numeric, predictive attributes and the target
    <BLANKLINE>
        :Attribute Information:
            - MedInc        median income in block group
            - HouseAge      median house age in block group
            - AveRooms      average number of rooms per household
            - AveBedrms     average number of bedrooms per household
            - Population    block group population
            - AveOccup      average number of household members
            - Latitude      block group latitude
            - Longitude     block group longitude
    <BLANKLINE>
        :Missing Attribute Values: None
    <BLANKLINE>
    This dataset was obtained from the StatLib repository.
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    <BLANKLINE>
    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of dollars ($100,000).
    <BLANKLINE>
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    <BLANKLINE>
    A household is a group of people residing within a home. Since the average
    number of rooms and bedrooms in this dataset are provided per household, these
    columns may take surprisingly large values for block groups with few households
    and many empty houses, such as vacation resorts.
    <BLANKLINE>
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    <BLANKLINE>
    .. topic:: References
    <BLANKLINE>
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297


It often helps to quickly visualize pieces of the data using histograms,
scatter plots, or other plot types. With matplotlib, let us show a
histogram of the target values: the median price in each neighborhood::

    >>> plt.hist(data.target)
    (array([...

.. image:: auto_examples/images/sphx_glr_plot_california_prediction_001.png
   :align: center
   :target: auto_examples/plot_california_prediction.html
   :scale: 70



Let's have a quick look to see if some features are more relevant than
others for our problem::

    >>> for index, feature_name in enumerate(data.feature_names):
    ...     plt.figure()
    ...     plt.scatter(data.data[feature_name], data.target)
    <Figure size...

.. image:: auto_examples/images/sphx_glr_plot_california_prediction_002.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_003.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_004.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_005.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_006.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_007.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_008.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html
.. image:: auto_examples/images/sphx_glr_plot_california_prediction_009.png
   :width: 32%
   :target: auto_examples/plot_california_prediction.html


This is a manual version of a technique called **feature selection**.

.. tip::

    Sometimes, in Machine Learning it is useful to use feature selection to
    decide which features are the most useful for a particular problem.
    Automated methods exist which quantify this sort of exercise of choosing
    the most informative features.

Predicting Home Prices: a Simple Linear Regression
--------------------------------------------------

Now we'll use ``scikit-learn`` to perform a simple linear regression on
the housing data. There are many possibilities of regressors to use. A
particularly simple one is ``LinearRegression``: this is basically a
wrapper around an ordinary least squares calculation. ::

    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    >>> from sklearn.linear_model import LinearRegression
    >>> clf = LinearRegression()
    >>> clf.fit(X_train, y_train)
    LinearRegression()
    >>> predicted = clf.predict(X_test)
    >>> expected = y_test
    >>> print("RMS: %s" % np.sqrt(np.mean((predicted - expected) ** 2)))
    RMS: 0.7...

.. image:: auto_examples/images/sphx_glr_plot_california_prediction_010.png
   :align: right
   :target: auto_examples/plot_california_prediction.html

We can plot the error: expected as a function of predicted::

    >>> plt.scatter(expected, predicted)
    <matplotlib.collections.PathCollection object at ...>

.. tip::

    The prediction at least correlates with the true price, though there are
    clearly some biases. We could imagine evaluating the performance of the
    regressor by, say, computing the RMS residuals between the true and
    predicted price. There are some subtleties in this, however, which we'll
    cover in a later section.

.. topic:: **Exercise: Gradient Boosting Tree Regression**
    :class: green

    There are many other types of regressors available in scikit-learn:
    we'll try a more powerful one here.

    **Use the GradientBoostingRegressor class to fit the housing data**.

    **hint** You can copy and paste some of the above code, replacing
    :class:`~sklearn.linear_model.LinearRegression` with
    :class:`~sklearn.ensemble.GradientBoostingRegressor`::

        from sklearn.ensemble import GradientBoostingRegressor
        # Instantiate the model, fit the results, and scatter in vs. out

    **Solution** The solution is found in :ref:`the code of this chapter <sphx_glr_packages_scikit-learn_auto_examples_plot_california_prediction.py>`



Measuring prediction performance
================================

A quick test on the K-neighbors classifier
------------------------------------------

Here we'll continue to look at the digits data, but we'll switch to the
K-Neighbors classifier.  The K-neighbors classifier is an instance-based
classifier.  The K-neighbors classifier predicts the label of
an unknown point based on the labels of the *K* nearest points in the
parameter space. ::

    >>> # Get the data
    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits()
    >>> X = digits.data
    >>> y = digits.target

    >>> # Instantiate and train the classifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> clf = KNeighborsClassifier(n_neighbors=1)
    >>> clf.fit(X, y)
    KNeighborsClassifier(...)

    >>> # Check the results using metrics
    >>> from sklearn import metrics
    >>> y_pred = clf.predict(X)

    >>> print(metrics.confusion_matrix(y_pred, y))
    [[178   0   0   0   0   0   0   0   0   0]
     [  0 182   0   0   0   0   0   0   0   0]
     [  0   0 177   0   0   0   0   0   0   0]
     [  0   0   0 183   0   0   0   0   0   0]
     [  0   0   0   0 181   0   0   0   0   0]
     [  0   0   0   0   0 182   0   0   0   0]
     [  0   0   0   0   0   0 181   0   0   0]
     [  0   0   0   0   0   0   0 179   0   0]
     [  0   0   0   0   0   0   0   0 174   0]
     [  0   0   0   0   0   0   0   0   0 180]]

Apparently, we've found a perfect classifier!  But this is misleading for
the reasons we saw before: the classifier essentially "memorizes" all the
samples it has already seen.  To really test how well this algorithm
does, we need to try some samples it *hasn't* yet seen.

This problem also occurs with regression models. In the following we
fit an other instance-based model named "decision tree" to the California
Housing price dataset we introduced previously::

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> data = fetch_california_housing(as_frame=True)
    >>> clf = DecisionTreeRegressor().fit(data.data, data.target)
    >>> predicted = clf.predict(data.data)
    >>> expected = data.target

    >>> plt.scatter(expected, predicted)
    <matplotlib.collections.PathCollection object at ...>
    >>> plt.plot([0, 50], [0, 50], '--k')
    [<matplotlib.lines.Line2D object at ...]

.. figure:: auto_examples/images/sphx_glr_plot_measuring_performance_001.png
   :align: right
   :target: auto_examples/plot_measuring_performance.html
   :width: 350

Here again the predictions are seemingly perfect as the model was able to
perfectly memorize the training set.

.. warning:: **Performance on test set**

   Performance on test set does not measure overfit (as described above)

A correct approach: Using a validation set
------------------------------------------

Learning the parameters of a prediction function and testing it on the
same data is a methodological mistake: a model that would just repeat the
labels of the samples that it has just seen would have a perfect score
but would fail to predict anything useful on yet-unseen data.

To avoid over-fitting, we have to define two different sets:

* a training set X_train, y_train which is used for learning the
  parameters of a predictive model

* a testing set X_test, y_test which is used for evaluating the fitted
  predictive model

In scikit-learn such a random split can be quickly computed with the
:func:`~sklearn.model_selection.train_test_split` function::

    >>> from sklearn import model_selection
    >>> X = digits.data
    >>> y = digits.target

    >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
    ...                                         test_size=0.25, random_state=0)

    >>> print("%r, %r, %r" % (X.shape, X_train.shape, X_test.shape))
    (1797, 64), (1347, 64), (450, 64)

Now we train on the training data, and test on the testing data::

    >>> clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)

    >>> print(metrics.confusion_matrix(y_test, y_pred))
    [[37  0  0  0  0  0  0  0  0  0]
     [ 0 43  0  0  0  0  0  0  0  0]
     [ 0  0 43  1  0  0  0  0  0  0]
     [ 0  0  0 45  0  0  0  0  0  0]
     [ 0  0  0  0 38  0  0  0  0  0]
     [ 0  0  0  0  0 47  0  0  0  1]
     [ 0  0  0  0  0  0 52  0  0  0]
     [ 0  0  0  0  0  0  0 48  0  0]
     [ 0  0  0  0  0  0  0  0 48  0]
     [ 0  0  0  1  0  1  0  0  0 45]]
    >>> print(metrics.classification_report(y_test, y_pred))
                  precision    recall  f1-score   support
    <BLANKLINE>
               0       1.00      1.00      1.00        37
               1       1.00      1.00      1.00        43
               2       1.00      0.98      0.99        44
               3       0.96      1.00      0.98        45
               4       1.00      1.00      1.00        38
               5       0.98      0.98      0.98        48
               6       1.00      1.00      1.00        52
               7       1.00      1.00      1.00        48
               8       1.00      1.00      1.00        48
               9       0.98      0.96      0.97        47
    <BLANKLINE>
        accuracy                           0.99       450
       macro avg       0.99      0.99      0.99       450
    weighted avg       0.99      0.99      0.99       450
    <BLANKLINE>

The averaged f1-score is often used as a convenient measure of the
overall performance of an algorithm.  It appears in the bottom row
of the classification report; it can also be accessed directly::

    >>> metrics.f1_score(y_test, y_pred, average="macro")
    0.991367...

The over-fitting we saw previously can be quantified by computing the
f1-score on the training data itself::

    >>> metrics.f1_score(y_train, clf.predict(X_train), average="macro")
    1.0

.. note::

   **Regression metrics** In the case of regression models, we
   need to use different metrics, such as explained variance.

Model Selection via Validation
------------------------------

.. tip::

    We have applied Gaussian Naives, support vectors machines, and
    K-nearest neighbors classifiers to the digits dataset. Now that we
    have these validation tools in place, we can ask quantitatively which
    of the three estimators works best for this dataset.

* With the default hyper-parameters for each estimator, which gives the
  best f1 score on the **validation set**?  Recall that hyperparameters
  are the parameters set when you instantiate the classifier: for
  example, the ``n_neighbors`` in ``clf =
  KNeighborsClassifier(n_neighbors=1)`` ::

    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.svm import LinearSVC

    >>> X = digits.data
    >>> y = digits.target
    >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
    ...                             test_size=0.25, random_state=0)

    >>> for Model in [GaussianNB(), KNeighborsClassifier(), LinearSVC(dual=False)]:
    ...     clf = Model.fit(X_train, y_train)
    ...     y_pred = clf.predict(X_test)
    ...     print('%s: %s' %
    ...           (Model.__class__.__name__, metrics.f1_score(y_test, y_pred, average="macro")))
    GaussianNB: 0.8...
    KNeighborsClassifier: 0.9...
    LinearSVC: 0.9...

* For each classifier, which value for the hyperparameters gives the best
  results for the digits data?  For :class:`~sklearn.svm.LinearSVC`, use
  ``loss='l2'`` and ``loss='l1'``.  For
  :class:`~sklearn.neighbors.KNeighborsClassifier` we use
  ``n_neighbors`` between 1 and 10. Note that
  :class:`~sklearn.naive_bayes.GaussianNB` does not have any adjustable
  hyperparameters. ::

    LinearSVC(loss='l1'): 0.930570687535
    LinearSVC(loss='l2'): 0.933068826918
    -------------------
    KNeighbors(n_neighbors=1): 0.991367521884
    KNeighbors(n_neighbors=2): 0.984844206884
    KNeighbors(n_neighbors=3): 0.986775344954
    KNeighbors(n_neighbors=4): 0.980371905382
    KNeighbors(n_neighbors=5): 0.980456280495
    KNeighbors(n_neighbors=6): 0.975792419414
    KNeighbors(n_neighbors=7): 0.978064579214
    KNeighbors(n_neighbors=8): 0.978064579214
    KNeighbors(n_neighbors=9): 0.978064579214
    KNeighbors(n_neighbors=10): 0.975555089773

  **Solution:** :ref:`code source <sphx_glr_packages_scikit-learn_auto_examples_plot_compare_classifiers.py>`


Cross-validation
----------------

Cross-validation consists in repeatedly splitting the data in pairs of
train and test sets, called 'folds'. Scikit-learn comes with a function
to automatically compute score on all these folds. Here we do
:class:`~sklearn.model_selection.KFold` with k=5. ::

    >>> clf = KNeighborsClassifier()
    >>> from sklearn.model_selection import cross_val_score
    >>> cross_val_score(clf, X, y, cv=5) #doctest: +ELLIPSIS
    array([0.947...,  0.955...,  0.966...,  0.980...,  0.963... ])

We can use different splitting strategies, such as random splitting::

    >>> from sklearn.model_selection import ShuffleSplit
    >>> cv = ShuffleSplit(n_splits=5)
    >>> cross_val_score(clf, X, y, cv=cv)
    array([...])

.. tip::

    There exists `many different cross-validation strategies
    <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`_
    in scikit-learn. They are often useful to take in account non iid
    datasets.

Hyperparameter optimization with cross-validation
-------------------------------------------------

Consider regularized linear models, such as *Ridge Regression*, which
uses l2 regularization, and *Lasso Regression*, which uses l1
regularization. Choosing their regularization parameter is important.

Let us set these parameters on the Diabetes dataset, a simple regression
problem. The diabetes data consists of 10 physiological variables (age,
sex, weight, blood pressure) measure on 442 patients, and an indication
of disease progression after one year::

    >>> from sklearn.datasets import load_diabetes
    >>> data = load_diabetes()
    >>> X, y = data.data, data.target
    >>> print(X.shape)
    (442, 10)

With the default hyper-parameters: we compute the cross-validation score::

    >>> from sklearn.linear_model import Ridge, Lasso

    >>> for Model in [Ridge, Lasso]:
    ...     model = Model()
    ...     print('%s: %s' % (Model.__name__, cross_val_score(model, X, y).mean()))
    Ridge: 0.4...
    Lasso: 0.3...

Basic Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We compute the cross-validation score as a function of alpha, the
strength of the regularization for :class:`~sklearn.linear_model.Lasso`
and :class:`~sklearn.linear_model.Ridge`. We choose 20 values of alpha
between 0.0001 and 1::

    >>> alphas = np.logspace(-3, -1, 30)

    >>> for Model in [Lasso, Ridge]:
    ...     scores = [cross_val_score(Model(alpha), X, y, cv=3).mean()
    ...               for alpha in alphas]
    ...     plt.plot(alphas, scores, label=Model.__name__)
    [<matplotlib.lines.Line2D object at ...

.. image:: auto_examples/images/sphx_glr_plot_linear_model_cv_001.png
   :align: left
   :target: auto_examples/plot_linear_model_cv.html
   :scale: 70


.. topic:: Question
   :class: green

   Can we trust our results to be actually useful?

Automatically Performing Grid Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`sklearn.grid_search.GridSearchCV` is constructed with an
estimator, as well as a dictionary of parameter values to be searched.
We can find the optimal parameters this way::

    >>> from sklearn.model_selection import GridSearchCV
    >>> for Model in [Ridge, Lasso]:
    ...     gscv = GridSearchCV(Model(), dict(alpha=alphas), cv=3).fit(X, y)
    ...     print('%s: %s' % (Model.__name__, gscv.best_params_))
    Ridge: {'alpha': 0.062101694189156162}
    Lasso: {'alpha': 0.01268961003167922}

Built-in Hyperparameter Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some models within scikit-learn, cross-validation can be performed
more efficiently on large datasets.  In this case, a cross-validated
version of the particular model is included.  The cross-validated
versions of :class:`~sklearn.linear_model.Ridge` and
:class:`~sklearn.linear_model.Lasso` are
:class:`~sklearn.linear_model.RidgeCV` and
:class:`~sklearn.linear_model.LassoCV`, respectively.  Parameter search
on these estimators can be performed as follows::

    >>> from sklearn.linear_model import RidgeCV, LassoCV
    >>> for Model in [RidgeCV, LassoCV]:
    ...     model = Model(alphas=alphas, cv=3).fit(X, y)
    ...     print('%s: %s' % (Model.__name__, model.alpha_))
    RidgeCV: 0.0621016941892
    LassoCV: 0.0126896100317

We see that the results match those returned by GridSearchCV

Nested cross-validation
~~~~~~~~~~~~~~~~~~~~~~~

How do we measure the performance of these estimators? We have used data
to set the hyperparameters, so we need to test on actually new data. We
can do this by running :func:`~sklearn.model_selection.cross_val_score`
on our CV objects. Here there are 2 cross-validation loops going on, this
is called *'nested cross validation'*::

    for Model in [RidgeCV, LassoCV]:
        scores = cross_val_score(Model(alphas=alphas, cv=3), X, y, cv=3)
        print(Model.__name__, np.mean(scores))


.. note::

    Note that these results do not match the best results of our curves
    above, and :class:`~sklearn.linear_model.LassoCV` seems to
    under-perform :class:`~sklearn.linear_model.RidgeCV`. The reason is
    that setting the hyper-parameter is harder for Lasso, thus the
    estimation error on this hyper-parameter is larger.

Unsupervised Learning: Dimensionality Reduction and Visualization
=================================================================

Unsupervised learning is applied on X without y: data without labels. A
typical use case is to find hidden structure in the data.

Dimensionality Reduction: PCA
-----------------------------

Dimensionality reduction derives a set of new artificial features smaller
than the original feature set. Here we'll use `Principal Component
Analysis (PCA)
<https://en.wikipedia.org/wiki/Principal_component_analysis>`__, a
dimensionality reduction that strives to retain most of the variance of
the original data. We'll use :class:`sklearn.decomposition.PCA` on the
iris dataset::

    >>> X = iris.data
    >>> y = iris.target

.. tip::

    :class:`~sklearn.decomposition.PCA` computes linear combinations of
    the original features using a truncated Singular Value Decomposition
    of the matrix X, to project the data onto a base of the top singular
    vectors.

::

    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=2, whiten=True)
    >>> pca.fit(X)
    PCA(n_components=2, whiten=True)

Once fitted, :class:`~sklearn.decomposition.PCA` exposes the singular
vectors in the ``components_`` attribute::

    >>> pca.components_
    array([[ 0.3..., -0.08...,  0.85...,  0.3...],
           [ 0.6...,  0.7..., -0.1..., -0.07...]])

Other attributes are available as well::

    >>> pca.explained_variance_ratio_
    array([0.92...,  0.053...])

Let us project the iris dataset along those first two dimensions:::

    >>> X_pca = pca.transform(X)
    >>> X_pca.shape
    (150, 2)

:class:`~sklearn.decomposition.PCA` ``normalizes`` and ``whitens`` the data, which means that the data
is now centered on both components with unit variance::

    >>> X_pca.mean(axis=0)
    array([...e-15,  ...e-15])
    >>> X_pca.std(axis=0, ddof=1)
    array([1.,  1.])

Furthermore, the samples components do no longer carry any linear
correlation::

    >>> np.corrcoef(X_pca.T)  # doctest: +SKIP
    array([[1.00000000e+00,   0.0],
           [0.0,   1.00000000e+00]])

With a number of retained components 2 or 3, PCA is useful to visualize
the dataset::

    >>> target_ids = range(len(iris.target_names))
    >>> for i, c, label in zip(target_ids, 'rgbcmykw', iris.target_names):
    ...     plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
    ...                 c=c, label=label)
    <matplotlib.collections.PathCollection ...

.. image:: auto_examples/images/sphx_glr_plot_pca_001.png
   :align: left
   :target: auto_examples/plot_pca.html
   :scale: 70

.. tip::

    Note that this projection was determined *without* any information
    about the labels (represented by the colors): this is the sense in
    which the learning is **unsupervised**. Nevertheless, we see that the
    projection gives us insight into the distribution of the different
    flowers in parameter space: notably, *iris setosa* is much more
    distinct than the other two species.


Visualization with a non-linear embedding: tSNE
-----------------------------------------------

For visualization, more complex embeddings can be useful (for statistical
analysis, they are harder to control). :class:`sklearn.manifold.TSNE` is
such a powerful manifold learning method. We apply it to the *digits*
dataset, as the digits are vectors of dimension 8*8 = 64. Embedding them
in 2D enables visualization::

    >>> # Take the first 500 data points: it's hard to see 1500 points
    >>> X = digits.data[:500]
    >>> y = digits.target[:500]

    >>> # Fit and transform with a TSNE
    >>> from sklearn.manifold import TSNE
    >>> tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0)
    >>> X_2d = tsne.fit_transform(X)

    >>> # Visualize the data
    >>> plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    <matplotlib.collections.PathCollection object at ...>


.. image:: auto_examples/images/sphx_glr_plot_tsne_001.png
   :align: left
   :target: auto_examples/plot_tsne.html
   :scale: 70


.. topic:: fit_transform

    As :class:`~sklearn.manifold.TSNE` cannot be applied to new data, we
    need to use its `fit_transform` method.

|

:class:`sklearn.manifold.TSNE` separates quite well the different classes
of digits even though it had no access to the class information.

.. raw:: html

    <div style="clear: both"></div>


.. topic:: Exercise: Other dimension reduction of digits
    :class: green

    :mod:`sklearn.manifold` has many other non-linear embeddings. Try
    them out on the digits dataset. Could you judge their quality without
    knowing the labels ``y``? ::

        >>> from sklearn.datasets import load_digits
        >>> digits = load_digits()
        >>> # ...

Parameter selection, Validation, and Testing
============================================

Hyperparameters, Over-fitting, and Under-fitting
------------------------------------------------

.. seealso::

    This section is adapted from `Andrew Ng's excellent
    Coursera course <https://www.coursera.org/course/ml>`__

The issues associated with validation and cross-validation are some of
the most important aspects of the practice of machine learning.
Selecting the optimal model for your data is vital, and is a piece of
the problem that is not often appreciated by machine learning
practitioners.

The central question is: **If our estimator is underperforming, how
should we move forward?**

-  Use simpler or more complicated model?
-  Add more features to each observed data point?
-  Add more training samples?

The answer is often counter-intuitive. In particular, **Sometimes using
a more complicated model will give worse results.** Also, **Sometimes
adding training data will not improve your results.** The ability to
determine what steps will improve your model is what separates the
successful machine learning practitioners from the unsuccessful.

Bias-variance trade-off: illustration on a simple regression problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. sidebar:: Code and notebook

   Python code and Jupyter notebook for this section are found
   :ref:`here
   <sphx_glr_packages_scikit-learn_auto_examples_plot_variance_linear_regr.py>`


Let us start with a simple 1D regression problem. This
will help us to easily visualize the data and the model, and the results
generalize easily to higher-dimensional datasets. We'll explore a simple
**linear regression** problem, with :mod:`sklearn.linear_model`.


.. include:: auto_examples/plot_variance_linear_regr.rst
    :start-after: We consider the situation where we have only 2 data point
    :end-before: **Total running time of the script:**


As we can see, the estimator displays much less variance. However it
systematically under-estimates the coefficient. It displays a biased
behavior.

This is a typical example of **bias/variance tradeof**: non-regularized
estimator are not biased, but they can display a lot of variance.
Highly-regularized models have little variance, but high bias. This bias
is not necessarily a bad thing: what matters is choosing the
tradeoff between bias and variance that leads to the best prediction
performance. For a specific dataset there is a sweet spot corresponding
to the highest complexity that the data can support, depending on the
amount of noise and of observations available.

Visualizing the Bias/Variance Tradeoff
--------------------------------------

.. tip::

    Given a particular dataset and a model (e.g. a polynomial), we'd like to
    understand whether bias (underfit) or variance limits prediction, and how
    to tune the *hyperparameter* (here ``d``, the degree of the polynomial)
    to give the best fit.

On a given data, let us fit a simple polynomial regression model with
varying degrees:

.. image:: auto_examples/images/sphx_glr_plot_bias_variance_001.png
   :align: center
   :target: auto_examples/plot_bias_variance.html

.. tip::

    In the above figure, we see fits for three different values of ``d``.
    For ``d = 1``, the data is under-fit. This means that the model is too
    simplistic: no straight line will ever be a good fit to this data. In
    this case, we say that the model suffers from high bias. The model
    itself is biased, and this will be reflected in the fact that the data
    is poorly fit. At the other extreme, for ``d = 6`` the data is over-fit.
    This means that the model has too many free parameters (6 in this case)
    which can be adjusted to perfectly fit the training data. If we add a
    new point to this plot, though, chances are it will be very far from the
    curve representing the degree-6 fit. In this case, we say that the model
    suffers from high variance. The reason for the term "high variance" is
    that if any of the input points are varied slightly, it could result in
    a very different model.

    In the middle, for ``d = 2``, we have found a good mid-point. It fits
    the data fairly well, and does not suffer from the bias and variance
    problems seen in the figures on either side. What we would like is a way
    to quantitatively identify bias and variance, and optimize the
    metaparameters (in this case, the polynomial degree d) in order to
    determine the best algorithm.

.. topic:: Polynomial regression with scikit-learn

   A polynomial regression is built by pipelining
   :class:`~sklearn.preprocessing.PolynomialFeatures`
   and a :class:`~sklearn.linear_model.LinearRegression`::

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> from sklearn.linear_model import LinearRegression
    >>> model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())


Validation Curves
~~~~~~~~~~~~~~~~~

Let us create a dataset like in the example above::

    >>> def generating_func(x, rng, err=0.5):
    ...     return rng.normal(10 - 1. / (x + 0.1), err)

    >>> # randomly sample more data
    >>> rng = np.random.default_rng(27446968)
    >>> x = rng.random(size=200)
    >>> y = generating_func(x, err=1., rng=rng)

.. image:: auto_examples/images/sphx_glr_plot_bias_variance_002.png
   :align: right
   :target: auto_examples/plot_bias_variance.html
   :scale: 60

Central to quantify bias and variance of a model is to apply it on *test
data*, sampled from the same distribution as the train, but that will
capture independent noise::

    >>> xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4)


.. raw:: html

    <div style="clear: both"></div>

**Validation curve** A validation curve consists in varying a model parameter
that controls its complexity (here the degree of the
polynomial) and measures both error of the model on training data, and on
test data (*eg* with cross-validation). The model parameter is then
adjusted so that the test error is minimized:

We use :func:`sklearn.model_selection.validation_curve` to compute train
and test error, and plot it::

    >>> from sklearn.model_selection import validation_curve

    >>> degrees = np.arange(1, 21)

    >>> model = make_pipeline(PolynomialFeatures(), LinearRegression())

    >>> # Vary the "degrees" on the pipeline step "polynomialfeatures"
    >>> train_scores, validation_scores = validation_curve(
    ...                 model, x[:, np.newaxis], y,
    ...                 param_name='polynomialfeatures__degree',
    ...                 param_range=degrees)

    >>> # Plot the mean train score and validation score across folds
    >>> plt.plot(degrees, validation_scores.mean(axis=1), label='cross-validation')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(degrees, train_scores.mean(axis=1), label='training')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.legend(loc='best')
    <matplotlib.legend.Legend object at ...>

.. image:: auto_examples/images/sphx_glr_plot_bias_variance_003.png
   :align: left
   :target: auto_examples/plot_bias_variance.html
   :scale: 60


This figure shows why validation is important. On the left side of the
plot, we have very low-degree polynomial, which under-fit the data. This
leads to a low explained variance for both the training set and the
validation set. On the far right side of the plot, we have a very high
degree polynomial, which over-fits the data. This can be seen in the fact
that the training explained variance is very high, while on the
validation set, it is low. Choosing ``d`` around 4 or 5 gets us the best
tradeoff.

.. tip::

    The astute reader will realize that something is amiss here: in the
    above plot, ``d = 4`` gives the best results. But in the previous plot,
    we found that ``d = 6`` vastly over-fits the data. What’s going on here?
    The difference is the **number of training points** used. In the
    previous example, there were only eight training points. In this
    example, we have 100. As a general rule of thumb, the more training
    points used, the more complicated model can be used. But how can you
    determine for a given model whether more training points will be
    helpful? A useful diagnostic for this are learning curves.

Learning Curves
~~~~~~~~~~~~~~~

A learning curve shows the training and validation score as a
function of the number of training points. Note that when we train on a
subset of the training data, the training score is computed using
this subset, not the full training set. This curve gives a
quantitative view into how beneficial it will be to add training
samples.

.. topic:: **Questions:**
   :class: green

   - As the number of training samples are increased, what do you expect
     to see for the training score? For the validation score?
   - Would you expect the training score to be higher or lower than the
     validation score? Would you ever expect this to change?


:mod:`scikit-learn` provides
:func:`sklearn.model_selection.learning_curve`::

    >>> from sklearn.model_selection import learning_curve
    >>> train_sizes, train_scores, validation_scores = learning_curve(
    ...     model, x[:, np.newaxis], y, train_sizes=np.logspace(-1, 0, 20))

    >>> # Plot the mean train score and validation score across folds
    >>> plt.plot(train_sizes, validation_scores.mean(axis=1), label='cross-validation')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(train_sizes, train_scores.mean(axis=1), label='training')
    [<matplotlib.lines.Line2D object at ...>]


.. figure:: auto_examples/images/sphx_glr_plot_bias_variance_004.png
   :align: left
   :target: auto_examples/plot_bias_variance.html
   :scale: 60

   For a ``degree=1`` model

Note that the validation score *generally increases* with a growing
training set, while the training score *generally decreases* with a
growing training set. As the training size
increases, they will converge to a single value.

From the above discussion, we know that ``d = 1`` is a high-bias
estimator which under-fits the data. This is indicated by the fact that
both the training and validation scores are low. When confronted
with this type of learning curve, we can expect that adding more
training data will not help: both lines converge to a
relatively low score.

|clear-floats|

**When the learning curves have converged to a low score, we have a
high bias model.**

A high-bias model can be improved by:

-  Using a more sophisticated model (i.e. in this case, increase ``d``)
-  Gather more features for each sample.
-  Decrease regularization in a regularized model.

Increasing the number of samples, however, does not improve a high-bias
model.

Now let's look at a high-variance (i.e. over-fit) model:

.. figure:: auto_examples/images/sphx_glr_plot_bias_variance_006.png
   :align: left
   :target: auto_examples/plot_bias_variance.html
   :scale: 60

   For a ``degree=15`` model


Here we show the learning curve for ``d = 15``. From the above
discussion, we know that ``d = 15`` is a **high-variance** estimator
which **over-fits** the data. This is indicated by the fact that the
training score is much higher than the validation score. As we add more
samples to this training set, the training score will continue to
decrease, while the cross-validation error will continue to increase, until they
meet in the middle.

|clear-floats|

**Learning curves that have not yet converged with the full training
set indicate a high-variance, over-fit model.**

A high-variance model can be improved by:

-  Gathering more training samples.
-  Using a less-sophisticated model (i.e. in this case, make ``d``
   smaller)
-  Increasing regularization.

In particular, gathering more features for each sample will not help the
results.

Summary on model selection
--------------------------

We’ve seen above that an under-performing algorithm can be due to two
possible situations: high bias (under-fitting) and high variance
(over-fitting). In order to evaluate our algorithm, we set aside a
portion of our training data for cross-validation. Using the technique
of learning curves, we can train on progressively larger subsets of the
data, evaluating the training error and cross-validation error to
determine whether our algorithm has high variance or high bias. But what
do we do with this information?

High Bias
~~~~~~~~~

If a model shows high **bias**, the following actions might help:

-  **Add more features**. In our example of predicting home prices, it
   may be helpful to make use of information such as the neighborhood
   the house is in, the year the house was built, the size of the lot,
   etc. Adding these features to the training and test sets can improve
   a high-bias estimator
-  **Use a more sophisticated model**. Adding complexity to the model
   can help improve on bias. For a polynomial fit, this can be
   accomplished by increasing the degree d. Each learning technique has
   its own methods of adding complexity.
-  **Use fewer samples**. Though this will not improve the
   classification, a high-bias algorithm can attain nearly the same
   error with a smaller training sample. For algorithms which are
   computationally expensive, reducing the training sample size can lead
   to very large improvements in speed.
-  **Decrease regularization**. Regularization is a technique used to
   impose simplicity in some machine learning models, by adding a
   penalty term that depends on the characteristics of the parameters.
   If a model has high bias, decreasing the effect of regularization can
   lead to better results.

High Variance
~~~~~~~~~~~~~

If a model shows **high variance**, the following actions might
help:

-  **Use fewer features**. Using a feature selection technique may be
   useful, and decrease the over-fitting of the estimator.
-  **Use a simpler model**. Model complexity and over-fitting go
   hand-in-hand.
-  **Use more training samples**. Adding training samples can reduce the
   effect of over-fitting, and lead to improvements in a high variance
   estimator.
-  **Increase Regularization**. Regularization is designed to prevent
   over-fitting. In a high-variance model, increasing regularization can
   lead to better results.

These choices become very important in real-world situations. For
example, due to limited telescope time, astronomers must seek a balance
between observing a large number of objects, and observing a large
number of features for each object. Determining which is more important
for a particular learning task can inform the observing strategy that
the astronomer employs.

A last word of caution: separate validation and test set
--------------------------------------------------------

Using validation schemes to determine hyper-parameters means that we are
fitting the hyper-parameters to the particular validation set. In the
same way that parameters can be over-fit to the training set,
hyperparameters can be over-fit to the validation set. Because of this,
the validation error tends to under-predict the classification error of
new data.

For this reason, it is recommended to split the data into three sets:

-  The **training set**, used to train the model (usually ~60% of the
   data)
-  The **validation set**, used to validate the model (usually ~20% of
   the data)
-  The **test set**, used to evaluate the expected error of the
   validated model (usually ~20% of the data)

Many machine learning practitioners do not separate test set and
validation set. But if your goal is to gauge the error of a model on
unknown data, using an independent test set is vital.

|

.. include:: auto_examples/index.rst
    :start-line: 1

.. seealso:: **Going further**

   * The `documentation of scikit-learn <https://scikit-learn.org>`__ is
     very complete and didactic.

   * `Introduction to Machine Learning with Python
     <https://shop.oreilly.com/product/0636920030515.do>`_,
     by Sarah Guido, Andreas Müller
     (`notebooks available here <https://github.com/amueller/introduction_to_ml_with_python>`_).
