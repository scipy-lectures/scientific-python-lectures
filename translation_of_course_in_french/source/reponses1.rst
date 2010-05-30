
Answers to Numpy's exercises
=============================


.. _answers1:

Array creation
---------------------

**Exercices** : create the following arrays in the easiest possible way
(and not element by element) ::

    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  2.]
     [ 1.  6.  1.  1.]]

    [[0 0 0 0 0]
     [2 0 0 0 0]
     [0 3 0 0 0]
     [0 0 4 0 0]
     [0 0 0 5 0]
     [0 0 0 0 6]]



..
   >>> import numpy as np

Answers ::

    >>> a = np.ones((4,4))
    >>> a[3,1] = 6
    >>> a[2,3] = 2
    >>> a
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  2.],
           [ 1.,  6.,  1.,  1.]])
    >>> b = np.diag(np.arange(1,7))
    >>> b = b[:,1:]
    >>> b
    array([[0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0],
           [0, 3, 0, 0, 0],
           [0, 0, 4, 0, 0],
           [0, 0, 0, 5, 0],
           [0, 0, 0, 0, 6]])

.. _stat_recherche:

Statistics of women's representation in French Research
-------------------------------------------------------------------

1. Get the following files ``organisms.txt`` and ``women_percentage.txt``
   (from the USB key of the training or
http://www.dakarlug.org/pat/scientifique/data/).

2. Create a ``data`` array by opening the ``women_percentage.txt`` file
   with ``np.loadtxt``. What is the shape of this array?

::

    >>> data = np.loadtxt('women_percentage.txt')
    >>> data.shape
    (21, 6)

3. Columns correspond to year 2006 to 2001. Create a ``years`` array with
   integers corresponding to these years.

::

    >>> years = np.arange(2006, 2000, -1)
    >>> years
    array([2006, 2005, 2004, 2003, 2002, 2001])

4. The different lines correspond to the research organisms whose names
   are stored in the ``organisms.txt`` file. Create a ``organisms`` array by
   opening this file. Beware that ``np.loadtxt`` creates float arrays by
   default, and it must be specified to use strings instead: ``organisms =
   np.loadtxt('organisms.txt', dtype=str)``

::

    >>> organisms = np.loadtxt('organismes.txt', dtype=str)
    >>> organisms
    array(['ADEME', 'BRGM', 'CEA', 'CEMAGREF', 'CIRAD', 'CNES', 'CNRS', 'CSTB',
           'IFREMER', 'INED', 'INERIS', 'INRA', 'INRETS', 'INRIA', 'INSERM',
           'IRD', 'IRSN', 'LCPC', 'ONERA', 'Pasteur', 'Universites'], 
          dtype='|S11')

5. Check that the number of lines of ``data`` equals the number of lines
   of the organisms.

..

    >>> data.shape[0] == organisms.shape[0]
    True

6. What is the maximal percentage of women in all organisms, for all
   years taken together?

..

    >>> data.max() # max on the whole array
    56.299999999999997

7. Create an array with the temporal mean of the percentage of women for
   each organism? (i.e. the mean of ``data`` along axis 1).

..

    >>> mn = data.mean(axis=1)
    >>> mn
    array([ 37.8       ,  23.16666667,  25.15      ,  24.51666667,
            24.38333333,  25.46666667,  30.88333333,  23.36666667,
            25.08333333,  52.53333333,  29.33333333,  37.58333333,
            31.86666667,  18.21666667,  50.16666667,  23.23333333,
            33.45      ,  15.18333333,  14.35      ,  49.86666667,  33.41666667])


8. Which organism had the highest percentage of women in 2004? (hint:
   use np.argmax)

..

    >>> np.nonzero(annees==2004)
    (array([2]),)
    >>> np.argmax(data[:,2])
    9
    >>> organisms[9]
    'INED'





9. Create a histogram of the percentage of women the different organisms
   in 2006 (hint: np.histogram, then matplotlib bar or plot for visulalization)


.. 

    >>> np.nonzero(annees==2006)
    (array([0]),)
    >>> hi = np.histogram(data[:,0])

.. sourcecode:: ipython

    In [88]: bar(hi[1][:-1], hi[0])

.. image:: barplot.png
   :align: center

10. Create an array that contains the organism where the highest women's
    percentage is found for the different years.


::

    >>> indices = np.argmax(data, axis=0)
    >>> indices
    array([ 9,  9,  9,  9,  9, 19])
    >>> organisms[indices]
    array(['INED', 'INED', 'INED', 'INED', 'INED', 'Pasteur'], 
          dtype='|S11')


