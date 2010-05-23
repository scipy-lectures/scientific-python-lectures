.. _pythonxy:

The Python(x,y) environment 
===========================

http://www.pythonxy.com/

Python(x,y) is a software that integrates a suite of standard
scientific-computing modules written in Python. Users of MS Windows may
find it easier to install once Python(x,y) than installing separately all
thses modules. Installing Python(x,y) is enough to start you off working
(and in particular, run all the examples of the tutorials).

Another solution is the Enthought Python Distribution (EPD), which is a
commercial product available for a larger choice of environments and with
commercial support.  


First steps with Python(x,y)
-----------------------------

Below the starting menu of Python(x,y) is displayed. You can choose
between running an application (a third-party software), or launching an 

Voici le menu de démarrage de Python(x,y). Il propose de choisir entre
lancer une application (un logiciel indépendant), ou de lancer une
"console interactive" Python, c'est-à-dire un interpréteur Python (shell
en anglais). Pour une session de calcul scientifique, nous choisissons de
lancer un interpréteur qui permettra d'exécuter nos différentes commandes
ainsi que des scripts. 

Parmi les différents interpréteurs disponibles, on peut conseiller
`Ipython`, qui est un interpréteur généraliste (utilisé ailleurs que dans
Python(x,y), et dans d'autres contextes que le calcul scientifique). 

.. image:: Python_xy_1.png
   :align: left

.. image:: Python_xy_2.png
   :align: center

On choisit d'ouvrir `Ipython` dans la "console 2", ce qui permet d'avoir
une représentation graphique plus esthétique que le shell Windows cmd. 

Une console s'ouvre alors, il faut attendre quelques secondes avant
d'avoir la main dans l'interpréteur car plusieurs modules sont importés
au début de la session.

Commençons par quelques commandes pour vérifier que tout marche::

    >>> a = "hello world"
    >>>> print a
    >>> # integers from 0 to 10
    >>> a = np.arange(10)
    >>> a 
    >>> from enthought.mayavi import mlab # 3D visualization
    >>> # Create data: coordinates of 3-D points picked at random
    >>> x, y, z = np.random.random((3,20))
    >>> mlab.points3d(x, y, z, z) 

Parmi les applications fournies par Python(x,y), on peut utiliser
l'éditeur de texte ``Scite`` (cf. flèche ci-dessous). ``Scite`` gère
correctement la coloration syntaxique de Python, l'indentation ou encore
l'exécution de scripts. Les fichiers python ont pour **extension .py**;
dès que Scite connaît l'extension il active la coloration syntaxique.   

.. image:: Python_scite.png
   :align: center


