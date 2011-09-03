========================================
scikit-learn: machine learning in Python
========================================

:author: Fabian Pedregosa

Machine learning is a rapidly-growing field with several machine
learning frameworks available for Python:

.. |mdp| image:: mdp.png
   :scale: 70   

.. |mlpy| image:: mlpy_logo.png
   :scale: 70   

.. |pymvpa| image:: pymvpa_logo.jpg
   :scale: 50   

.. |orange| image:: orange-logo-w.png
   :scale: 70

.. |skl| image:: scikit-learn-logo.png
   :scale: 40

.. only:: html

    .. centered:: |mdp|  |mlpy| |pymvpa|  |orange| |skl| 

.. only:: latex 

    |mdp|  |mlpy|
    
    |orange| |skl|


.. topic:: Prerequisites

    * Numpy, Scipy
    * IPython
    * matplotlib
    * scikit-learn (http://scikit-learn.sourceforge.net)


.. contents:: Chapters contents
   :local:
   :depth: 2


Loading an example dataset
==========================

.. raw:: html

    <div style='float: right; margin: 35px;'>

.. image:: images/Virginia_Iris.png
   :align: right
   :alt: Photo of Iris Virginia

.. raw:: html

    </div>


First we will load some data to play with. The data we will use is a
very simple flower database known as the Iris dataset.

We have 150 observations of the iris flower specifying some
measurements: sepal length, sepal width, petal length and petal width
together with its subtype: Iris Setosa, Iris Versicolour, Iris
Virginica.

.. for now, a dataset is just a matrix of floating-point numbers,
.. (together with a class value).

To load the dataset into a Python object:


::

  >>> from scikits.learn import datasets
  >>> iris = datasets.load_iris()

This data is stored in the ``.data`` member, which
is a ``(n_samples, n_features)`` array.

    >>> iris.data.shape
    (150, 4)

The class of each observation is stored in the ``.target`` attribute of the
dataset. This is an integer 1D array of length ``n_samples``:

    >>> iris.target.shape
    (150,)
    >>> import numpy as np
    >>> np.unique(iris.target)
    [0, 1, 2]


.. topic:: An example of reshaping data: the digits dataset

    .. image:: digits_first_image.png
        :scale: 50
        :align: right

    The digits dataset is made of 1797 images, where each one is a 8x8
    pixel image representing a hand-written digit ::

        >>> digits = datasets.load_digits()
        >>> digits.images.shape
        (1797, 8, 8)
        >>> import pylab as pl
        >>> pl.imshow(digits.images[0], cmap=pl.cm.gray_r) #doctest: +ELLIPSIS
        <matplotlib.image.AxesImage object at ...>

    To use this dataset with the scikit, we transform each 8x8 image
    into a vector of length 64 ::

        >>> data = digits.images.reshape((digits.images.shape[0], -1))




Learning and Predicting
+++++++++++++++++++++++

Now that we've got some data, we would like to learn from it and
predict on new one. In ``scikit-learn``, we learn from existing
data by creating an ``estimator`` and calling its ``fit(X, Y)`` method.

    >>> from scikits.learn import svm
    >>> clf = svm.LinearSVC()
    >>> clf.fit(iris.data, iris.target) # learn form the data

Once we have learned from the data, we can access the parameters of
the model:

    >>> clf.coef_
    ...

And it can be used to predict the most likely outcome on unseen data:

    >>> clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
    array([0], dtype=int32)



Classification
===================


k-Nearest neighbors classifier
++++++++++++++++++++++++++++++

The simplest possible classifier is the nearest neighbor: given a new
observation, take the label of the closest learned observation.

.. image:: iris_knn.png
   :scale: 90
   :align: right

Internally uses the BallTree algorithm.

**KNN (k nearest neighbors) classification example**:

::

    >>> # Create and fit a nearest-neighbor classifier
    >>> from scikits.learn import neighbors
    >>> knn = neighbors.NeighborsClassifier()
    >>> knn.fit(iris.data, iris.target)
    NeighborsClassifier(n_neighbors=5, leaf_size=20, algorithm='auto')
    >>> knn.predict([[0.1, 0.2, 0.3, 0.4]])
    array([0])


.. topic:: Training set and testing set

   When experimenting with learning algorithm, it is important not to
   test the prediction of an estimator on the data used to fit the
   estimator.

   ::

       >>> perm = np.random.permutation(iris.target.size)
       >>> iris.data = iris.data[perm]
       >>> iris.target = iris.target[perm]
       >>> knn.fit(iris.data[:100], iris.target[:100]
       >>> knn.score(iris.data[100:], iris.target[100:])



Support vector machines (SVMs) for classification
+++++++++++++++++++++++++++++++++++++++++++++++++

Linear Support Vector Machines
------------------------------

SVMs try to build a plane maximizing the margin between the two
classes. It selects a subset of the input, called the support vectors,
which are the observations closest to the separating plane.


.. image:: svm_margin.png
   :align: right 
   :scale: 80


.. Regularization is set by the `C` parameter: with small `C`
.. give (regularized problem) the margin is computed only on the
.. observation close to the separating plane; with large `C` all the
.. observations are used.


::

    >>> from scikits.learn import svm
    >>> svc = svm.SVC(kernel='linear')
    >>> svc.fit(iris.data, iris.target)
    SVC(kernel='linear', C=1.0, probability=False, degree=3, coef0=0.0, tol=0.001,
      shrinking=True, gamma=0.0)

There are several support vector machine implementations in
scikit-learn. The most used ones are ``svm.SVC``, ``svm.NuSVC`` and ``svm.LinearSVC``. 

.. topic:: **Excercise**
   :class: green

   Train an ``svm.SVC`` on the digits dataset. Leave out the
   last 10% and test prediction performance on these observations.



Using kernels
--------------

Classes are not always separable by a hyper-plane, thus it would be
desirable to have a decision function that is not linear but that may
be for instance polynomial or exponential:


.. |svm_kernel_linear| image:: svm_kernel_linear.png
   :scale: 65

.. |svm_kernel_poly| image:: svm_kernel_poly.png
   :scale: 65

.. |svm_kernel_rbf| image:: svm_kernel_rbf.png
   :scale: 65

.. rst-class:: centered

  .. list-table::

     *

       - **Linear kernel**

       - **Polynomial kernel**

       - **RBF kernel (Radial Basis Function)**

     *

       - |svm_kernel_linear|

       - |svm_kernel_poly|

       - |svm_kernel_rbf|

     *

       - ::

            >>> svc = svm.SVC(kernel='linear')

       - ::

            >>> svc = svm.SVC(kernel='poly',
            ...               degree=3)
            >>> # degree: polynomial degree

       - ::

            >>> svc = svm.SVC(kernel='rbf')
            >>> # gamma: inverse of size of
            >>> # radial kernel


.. topic:: **Exercise**
   :class: green

   Which of the kernels noted above has a better prediction
   performance on the digits dataset ?

   .. toctree::

        digits_classification_exercise



Clustering: grouping observations together
==========================================

Given the iris dataset, if we knew that there were 3 types of Iris,
but did not have access to their labels: we could try a **clustering
task**: split the observations into groups called *clusters*.



K-means clustering
++++++++++++++++++

The simplest clustering algorithm is the k-means.

::

    >>> from scikits.learn import cluster, datasets
    >>> iris = datasets.load_iris()
    >>> k_means = cluster.KMeans(k=3)
    >>> k_means.fit(iris.data) # doctest: +ELLIPSIS
    KMeans(verbose=0, k=3, max_iter=300, init='k-means++',...
    >>> print k_means.labels_[::10]
    [1 1 1 1 1 0 0 0 0 0 2 2 2 2 2]
    >>> print iris.target[::10]
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]

.. |cluster_iris_truth| image:: cluster_iris_truth.png
   :scale: 77

.. |cluster_iris_kmeans| image:: k_means_iris_3.png
    :scale: 80

.. |k_means_iris_8| image:: k_means_iris_8.png
   :scale: 77


.. list-table::
    :class: centered

    *
        - |cluster_iris_truth|

        - |cluster_iris_kmeans|

        - |k_means_iris_8|


    *
        - **Ground truth**

        - **K-means (3 clusters)**

        - **K-means (8 clusters)**



.. |lena| image:: lena.png
   :scale: 50

.. |lena_regular| image:: lena_regular.png
   :scale: 50

.. |lena_compressed| image:: lena_compressed.png
   :scale: 50


.. topic:: **Application to Image Compression**

    Clustering can be seen as a way of choosing a small number of
    observations from the information. For instance, this can be used
    to posterize an image (conversion of a continuous gradation of
    tone to several regions of fewer tones)::

    >>> import scipy as sp
    >>> lena = sp.lena().astype(np.float32)
    >>> X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
    >>> k_means = cluster.KMeans(k=5)
    >>> k_means.fit(X)
    >>> values = k_means.cluster_centers_.squeeze()
    >>> labels = k_means.labels_
    >>> lena_compressed = np.choose(labels, values)
    >>> lena_compressed.shape = lena.shape

    .. list-table::
      :class: centered

      *
        - |lena|

        - |lena_compressed|

      *

        - Raw image

        - K-means quantization



Dimension Reduction with Principal Component Analysis
=====================================================



.. |pca_3d_axis| image:: pca_3d_axis.jpg
   :scale: 70

.. |pca_3d_aligned| image:: pca_3d_aligned.jpg
   :scale: 70

.. rst-class:: centered

   |pca_3d_axis| |pca_3d_aligned|


The cloud of points spanned by the observations above is very flat in
one direction, so that one feature can almost be exactly computed
using the 2 other. PCA finds the directions in which the data is not
*flat* and it can reduce the dimensionality of the data by projecting
on a subspace.


.. warning::

    Depending on your version of scikit-learn PCA will be in module
    ``decomposition`` or ``pca``.

>>> from scikits.learn import decomposition
>>> pca = decomposition.PCA(n_components=2)
>>> pca.fit(iris.data)
PCA(copy=True, n_components=2, whiten=False)
>>> X = pca.transform(iris.data)

Now we can visualize the (transformed) iris dataset!

>>> import pylab as pl
>>> pl.scatter(X[:, 0], X[:, 1], c=iris.target)
>>> pl.show()

.. image:: pca_iris.png
   :scale: 50
   :align: center



PCA is not just useful for visualization of high dimensional
datasets. It can also be used as a preprocessing step to help speed up
supervised methods that are not computationally efficient with high
dimensions.



Putting it all together: face recognition
=========================================

An example showcasing face recognition using Principal Component
Analysis for dimension reduction and Support Vector Machines for
classification.

.. image:: faces.png
   :align: center
   :scale: 70


.. sourcecode:: python

    
    """
    Stripped-down version of the face recognition example by Olivier Grisel
    
    http://scikit-learn.sourceforge.net/dev/auto_examples/applications/face_recognition.html
    
    ## original shape of images: 50, 37
    """
    import numpy as np
    import pylab as pl
    from scikits.learn import cross_val, datasets, decomposition, svm
    
    # ..
    # .. load data ..
    lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    perm = np.random.permutation(lfw_people.target.size)
    lfw_people.data = lfw_people.data[perm]
    lfw_people.target = lfw_people.target[perm]
    faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
    train, test = iter(cross_val.StratifiedKFold(lfw_people.target, k=4)).next()
    X_train, X_test = faces[train], faces[test]
    y_train, y_test = lfw_people.target[train], lfw_people.target[test]
    
    # ..
    # .. dimension reduction ..
    pca = decomposition.RandomizedPCA(n_components=150, whiten=True)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # ..
    # .. classification ..
    clf = svm.SVC(C=5., gamma=0.001)
    clf.fit(X_train_pca, y_train)

    # ..
    # .. predict on new images ..
    for i in range(10):
        print lfw_people.target_names[clf.predict(X_test_pca[i])[0]]
        _ = pl.imshow(X_test[i].reshape(50, 37), cmap=pl.cm.gray)
        _ = raw_input()
    



Full code: :download:`faces.py`



Linear model: from regression to sparsity
==========================================

.. topic:: Diabetes dataset

    The diabetes dataset consists of 10 physiological variables (age,
    sex, weight, blood pressure) measure on 442 patients, and an
    indication of disease progression after one year::

        >>> diabetes = datasets.load_diabetes()
        >>> diabetes_X_train = diabetes.data[:-20]
        >>> diabetes_X_test  = diabetes.data[-20:]
        >>> diabetes_y_train = diabetes.target[:-20]
        >>> diabetes_y_test  = diabetes.target[-20:]
    
    The task at hand is to predict disease prediction from physiological
    variables. 


Sparse models
+++++++++++++

To improve the conditioning of the problem (uninformative variables,
mitigate the curse of dimensionality, as a feature selection
preprocessing, etc.), it would be interesting to select only the
informative features and set non-informative ones to 0. This
penalization approach, called **Lasso**, can set some coefficients to
zero.  Such methods are called **sparse method**, and sparsity can be
seen as an application of Occam's razor: prefer simpler models to
complex ones.

:: 

    >>> from scikits.learn import linear_model
    >>> regr = linear_model.Lasso(alpha=.3)
    >>> regr.fit(diabetes_X_train, diabetes_y_train)
    >>> regr.coef_ # very sparse coefficients
    array([   0.        ,   -0.        ,  497.34075682,  199.17441034,
             -0.        ,   -0.        , -118.89291545,    0.        ,
            430.9379595 ,    0.        ])
    >>> regr.score(diabetes_X_test, diabetes_y_test)
    0.55108354530029802

being the score very similar to linear regression (Least Squares)::

    >>> lin = linear_model.LinearRegression()
    >>> lin.fit(diabetes_X_train, diabetes_y_train)
    LinearRegression(fit_intercept=True, normalize=False, overwrite_X=False)
    >>> lin.score(diabetes_X_test, diabetes_y_test)
    0.58507530226905724

.. topic:: **Different algorithms for a same problem**

    Different algorithms can be used to solve the same mathematical
    problem. For instance the `Lasso` object in the `scikits.learn`
    solves the lasso regression using a *coordinate descent* method, that
    is efficient on large datasets. However, the `scikits.learn` also
    provides the `LassoLARS` object, using the *LARS* which is very
    efficient for problems in which the weight vector estimated is very
    sparse, that is problems with very few observations.


============================================================
Model selection: choosing estimators and their parameters
============================================================


Grid-search and cross-validated estimators
============================================

Grid-search
+++++++++++

The scikits.learn provides an object that, given data, computes the score
during the fit of an estimator on a parameter grid and chooses the
parameters to maximize the cross-validation score. This object takes an
estimator during the construction and exposes an estimator API::

    >>> from scikits.learn import svm, grid_search
    >>> gammas = np.logspace(-6, -1, 10)
    >>> svc = svm.SVC()
    >>> clf = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), 
    ...                    n_jobs=-1)
    >>> clf.fit(digits.data[:1000], digits.target[:1000]) # doctest: +ELLIPSIS
    GridSearchCV(n_jobs=-1, ...)
    >>> clf.best_score
    0.98899798001594419
    >>> clf.best_estimator.gamma
    0.00059948425031894088


By default the `GridSearchCV` uses a 3-fold cross-validation. However, if
it detects that a classifier is passed, rather than a regressor, it uses
a stratified 3-fold.



Cross-validated estimators
++++++++++++++++++++++++++

Cross-validation to set a parameter can be done more efficiently on an
algorithm-by-algorithm basis. This is why, for certain estimators, the
scikits.learn exposes "CV" estimators, that set their parameter
automatically by cross-validation::

    >>> from scikits.learn import linear_model, datasets
    >>> lasso = linear_model.LassoCV()
    >>> diabetes = datasets.load_diabetes()
    >>> X_diabetes = diabetes.data
    >>> y_diabetes = diabetes.target
    >>> lasso.fit(X_diabetes, y_diabetes)
    >>> # The estimator chose automatically its lambda:
    >>> lasso.alpha
    0.0075421928471338063

These estimators are called similarly to their counterparts, with 'CV'
appended to their name.

.. topic:: **Exercise**
   :class: green

   On the diabetes dataset, find the optimal regularization parameter
   alpha.





 

