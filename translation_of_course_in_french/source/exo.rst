Réponses au cas d'application : infondus dans un verre
=======================================================

.. image:: MV_HFV_012.jpg
   :align: center 
   :width: 600px


1. Ouvrir l'image MV_HFV_012.jpg. On cherche à déterminer la fraction du
matériau occupée par des grains infondus (gris foncés), du verre (gris 
clair), et des bulles (noir). On veut aussi estimer la taille typique des
grains de sable, leur nombre, les voisinages entre grains, etc.

::

    >>> dat = imread('MV_HFV_012.jpg')

2. Enlever le bandeau avec les informations de mesure.

::
    
    >>> dat = dat[60:]

3. Filtrer légèrement l'image avec un filtre médian afin d'affiner 
l'histogramme des niveaux d'intensité. Vérifier sur les histogrammes.

::

    >>> filtdat = ndimage.median_filter(dat, size=(7,7))
    >>> hi_dat = np.histogram(dat, bins=np.arange(256))
    >>> hi_filtdat = np.histogram(filtdat, bins=np.arange(256))

.. image:: exo_histos.png
   :align: center 

4. A partir de l'image filtrée, déterminer des seuils permettant de
définir un masque pour les pixels du sable, un pour le verre et un pour
les bulles.

::

    >>> void = filtdat <= 50
    >>> sand = np.logical_and(filtdat>50, filtdat<=114)
    >>> glass = filtdat > 114

5. Afficher une image où les trois phases sont coloriées chacune dans une
couleur différente.

::
    >>> phases = void.astype(np.int) + 2*glass.astype(np.int) +\
            3*sand.astype(np.int)

.. image:: three_phases.png
   :align: center 

6. Utiliser la morphologie mathématique pour nettoyer les différentes
phases.

::
    >>> sand_op = ndimage.binary_opening(sand, iterations=2)
    >>> sand_labels, sand_nb = ndimage.label(sand_op)
    >>> sand_areas = np.array(ndimage.sum(sand_op, sand_labels,\
    ...     np.arange(sand_labels.max()+1)))
    >>> mask = sand_areas>100
    >>> remove_small_sand = mask[sand_labels]

.. image:: sands.png
   :align: center 

