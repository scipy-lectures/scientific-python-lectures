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
----------------

Let's create some simple data::

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
predominant class.

Let's try it out on our iris classification problem::

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
regression one::

    # Create some simple data
    import numpy as np
    np.random.seed(0)
    X = np.random.random(size=(20, 1))
    y = 3 * X[:, 0] + 2 + np.random.normal(size=20)
    
    # Fit a linear regression to it
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    print("Model coefficient: %.5f, and intercept: %.5f"
          % (model.coef_, model.intercept_))
    
    # Plot the data and the model prediction
    X_test = np.linspace(0, 1, 100)[:, np.newaxis]
    y_test = model.predict(X_test)
    import pylab as pl
    plt.plot(X[:, 0], y, 'o')
    plt.plot(X_test[:, 0], y_test)

A recap on Scikit-learn's estimator interface
----------------------------------------------

Scikit-learn strives to have a uniform interface across all methods, and
we’ll see examples of these below. Given a scikit-learn *estimator*
object named ``model``, the following methods are available:

-  Available in **all Estimators**
-  ``model.fit()`` : fit training data. For supervised learning
   applications, this accepts two arguments: the data ``X`` and the
   labels ``y`` (e.g. ``model.fit(X, y)``). For unsupervised learning
   applications, this accepts only a single argument, the data ``X``
   (e.g. ``model.fit(X)``).
-  Available in **supervised estimators**
-  ``model.predict()`` : given a trained model, predict the label of a
   new set of data. This method accepts one argument, the new data
   ``X_new`` (e.g. ``model.predict(X_new)``), and returns the learned
   label for each object in the array.
-  ``model.predict_proba()`` : For classification problems, some
   estimators also provide this method, which returns the probability
   that a new observation has each categorical label. In this case, the
   label with the highest probability is returned by
   ``model.predict()``.
-  ``model.score()`` : for classification or regression problems, most
   (all?) estimators implement a score method. Scores are between 0 and
   1, with a larger score indicating a better fit.
-  Available in **unsupervised estimators**
-  ``model.transform()`` : given an unsupervised model, transform new
   data into the new basis. This also accepts one argument ``X_new``,
   and returns the new representation of the data based on the
   unsupervised model.
-  ``model.fit_transform()`` : some estimators implement this method,
   which more efficiently performs a fit and a transform on the same
   input data.

Regularization: what it is and why it is necessary
----------------------------------------------------

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

Regularization is ubiquitous in machine learning. Most scikit-learn
estimators have a parameter to tune the amount of regularization. For
instance, with k-NN, it is 'k', the number of nearest neighbors used to
make the decision. k=1 amounts to no regularization: 0 error on the
training set, whereas large k will push toward smoother decision
boundaries in the feature space.

Exercise: Interactive Demo on linearly separable data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the **svm\_gui.py** file in the repository:
https://github.com/GaelVaroquaux/sklearn\_ensae\_course

--------------
