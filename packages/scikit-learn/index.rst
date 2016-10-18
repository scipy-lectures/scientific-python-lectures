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
dimensions at a time using a scatter plot: 

.. image:: auto_examples/images/sphx_glr_plot_iris_scatter_001.png
   :align: center
   :target: auto_examples/plot_iris_scatter.html

.. topic:: **Excercise**:
    :class: green
   
    Can you choose 2 features to find a plot where it is easier to
    seperate the different classes of irises?

    **Hint**: click on the figure above to see the code that generates it,
    and modify this code.


Basic principles of machine learning with scikit-learn
======================================================

Introducing the scikit-learn estimator object
----------------------------------------------

Every algorithm is exposed in scikit-learn via an ''Estimator'' object.
For instance a linear regression is::

    >>> from sklearn.linear_model import LinearRegression

**Estimator parameters**: All the parameters of an estimator can be set
when it is instantiated::

    >>> model = LinearRegression(normalize=True)
    >>> print(model.normalize)
    True
    >>> print(model)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)

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
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)

**Estimated parameters**: When data is fitted with an estimator,
parameters are estimated from the data at hand. All the estimated
parameters are attributes of the estimator object ending by an
underscore::

    >>> model.coef_
    array([ 1.])

Supervised Learning: Classification and regression
---------------------------------------------------

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
   Prize <http://en.wikipedia.org/wiki/Netflix_prize>`__).

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
    :end-before: # plot the results

.. figure:: auto_examples/images/sphx_glr_plot_linear_regression_001.png
   :align: center
   :target: auto_examples/plot_linear_regression.html

   A plot of a simple linear regression.

A recap on Scikit-learn's estimator interface
----------------------------------------------

Scikit-learn strives to have a uniform interface across all methods, and
we’ll see examples of these below. Given a scikit-learn *estimator*
object named ``model``, the following methods are available:

* Available in **all Estimators**

  - ``model.fit()`` : fit training data. For supervised learning
    applications, this accepts two arguments: the data ``X`` and the
    labels ``y`` (e.g. ``model.fit(X, y)``). For unsupervised learning
    applications, this accepts only a single argument, the data ``X``
    (e.g. ``model.fit(X)``).
* Available in **supervised estimators**

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
* Available in **unsupervised estimators**

  - ``model.transform()`` : given an unsupervised model, transform new
    data into the new basis. This also accepts one argument ``X_new``,
    and returns the new representation of the data based on the
    unsupervised model.
  - ``model.fit_transform()`` : some estimators implement this method,
    which more efficiently performs a fit and a transform on the same
    input data.

Regularization: what it is and why it is necessary
----------------------------------------------------

Prefering simpler models
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

.. figure:: auto_examples/images/sphx_glr_plot_polynomial_regression_002.png
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
   `here <sphx_glr_packages_scikit-learn_auto_examples_plot_digits_simple_classif.py>`_


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
straightforward one, Principal Component Analysis (PCA).

PCA seeks orthogonal linear combinations of the features which show the
greatest variance, and as such, can help give you a good idea of the
structure of the data set. Here we'll use ``RandomizedPCA``, because
it's faster for large ``N``::

    >>> from sklearn.decomposition import RandomizedPCA
    >>> pca = RandomizedPCA(n_components=2)
    >>> proj = pca.fit_transform(digits.data)
    >>> plt.scatter(proj[:, 0], proj[:, 1], c=digits.target)
    >>> plt.colorbar() # doctest: +SKIP

.. image:: auto_examples/images/sphx_glr_plot_digits_simple_classif_002.png
   :align: center 
   :target: auto_examples/plot_digits_simple_classif.html

.. topic:: **Question**
   
    Given these projections of the data, which numbers do you think a
    classifier might have trouble distinguishing?

Gaussian Naive Bayes Classification
-----------------------------------

For most classification problems, it's nice to have a simple, fast,
go-to method to provide a quick baseline classification. If the simple
and fast method is sufficient, then we don't have to waste CPU cycles on
more complex models. If not, we can use the results of the simple method
to give us clues about our data.

One good method to keep in mind is Gaussian Naive Bayes. It fits a
Gaussian distribution to each training label independantly on each
feature, and uses this to quickly give a rough classification. It is
generally not sufficiently accurate for real-world data, but can perform
surprisingly well, for instance on text data::

    from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import train_test_split

    # split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
    
    # train the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # use the model to predict the labels of the test data
    predicted = clf.predict(X_test)
    expected = y_test

.. image:: auto_examples/images/sphx_glr_plot_digits_simple_classif_002.png
   :align: center 
   :target: auto_examples/plot_digits_simple_classif.html

As above, we can plot the digits with the predicted labels to get an idea of
how well the classification is working:

.. topic:: **Question**

    Why did we split the data into training and validation sets?

Quantitative Measurement of Performance
---------------------------------------

We'd like to measure the performance of our estimator without having to
resort to plotting examples. A simple method might be to simply compare
the number of matches::

    >>> matches = (predicted == expected)
    >>> print(matches.sum())
    397
    >>> print(len(matches))
    450
    >>> matches.sum() / float(len(matches))
    0.88222222222222224



We see that nearly 1500 of the 1800 predictions match the input. But
there are other more sophisticated metrics that can be used to judge the
performance of a classifier: several are available in the
``sklearn.metrics`` submodule.

One of the most useful metrics is the ``classification_report``, which
combines several measures and prints a table with the results::

    >>> from sklearn import metrics
    >>> print(metrics.classification_report(expected, predicted))
                 precision    recall  f1-score   support
    
              0       1.00      0.98      0.99        44
              1       0.84      0.82      0.83        39
              2       0.93      0.84      0.88        45
              3       0.91      0.83      0.87        48
              4       0.93      0.89      0.91        47
              5       0.94      0.88      0.91        51
              6       0.95      1.00      0.98        40
              7       0.78      1.00      0.88        46
              8       0.68      0.92      0.78        39
              9       0.95      0.69      0.80        51
    
    avg / total       0.89      0.88      0.88       450
    


Another enlightening metric for this sort of multi-label classification
is a *confusion matrix*: it helps us visualize which labels are being
interchanged in the classification errors::

    >>> print(metrics.confusion_matrix(expected, predicted))
    [[43  0  0  0  0  0  0  1  0  0]
     [ 0 32  2  0  0  0  1  1  2  1]
     [ 0  1 38  0  1  0  0  0  5  0]
     [ 0  0  1 40  0  2  0  1  3  1]
     [ 0  0  0  0 42  1  1  3  0  0]
     [ 0  0  0  2  1 45  0  2  1  0]
     [ 0  0  0  0  0  0 40  0  0  0]
     [ 0  0  0  0  0  0  0 46  0  0]
     [ 0  2  0  0  0  0  0  1 36  0]
     [ 0  3  0  2  1  0  0  4  6 35]]


We see here that in particular, the numbers 1, 2, 3, and 9 are often
being labeled 8.
