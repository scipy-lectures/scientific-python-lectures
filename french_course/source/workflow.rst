Les briques de l'informatique scientifique avec Python
========================================================

* **Python**, un langage de programmation généraliste et moderne.  

    * Le langage Python : types de données (``string``, ``int``), contrôle
      de flux, conteneurs (listes, dictionnaires), patterns, etc.

    * Les modules de la librairie standard

    * Un grand nombre d'autres modules spécialisés ou d'applications :
      protocoles web, frameworks web, etc. ... et calcul scientifique.

    * Outils de développement (tests automatisés, génération de
      documentation, etc.)

* **IPython**, un interpréteur avec des fonctionnalités avancées.

  http://ipython.scipy.org/moin/
 
.. image:: snapshot_ipython.png
      :align: center

* **Numpy** : fournit l'objet tableau (de données) et les routines pour
  manipuler ces tableaux::

    >>> import numpy as np
    >>> t = np.arange(10)
    >>> t
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> print t 
    [0 1 2 3 4 5 6 7 8 9]
    >>> signal = np.sin(t)

  http://www.scipy.org/

.. 
    >>> np.random.seed(4)

* **Scipy** : routines haut-niveau de traitement de données.
  Optimisation, régression, interpolation, etc::

    >>> import numpy as np
    >>> import scipy 
    >>> t = np.arange(10)
    >>> t
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> signal = t**2 + 2*t + 2+ 1.e-2*np.random.random(10)
    >>> scipy.polyfit(t, signal, 2)
    array([ 1.00001151,  1.99920674,  2.00902748])

  http://www.scipy.org/

* **Matplotlib** : visualisation 2-D, figures "publication-ready"

  http://matplotlib.sourceforge.net/

.. image:: random_c.jpg
      :height: 300px

.. image:: hexbin_demo.png
      :height: 300px
  

* **Mayavi** : visualisation 3-D
  
  http://code.enthought.com/projects/mayavi/

.. image:: example_surface_from_irregular_data.jpg
      :align: center    

