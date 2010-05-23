L'informatique scientifique : qu'est-ce que c'est ?
=====================================================

..
    .. image:: phd053104s.png
      :align: center

Les besoins du scientifique
-----------------------------

* Acquérir des données (simulations, contrôle d'expérience)

* Manipuler et traiter ces données.

* Visualiser les résultats... pour comprendre ce qu'on fait !

* Communiquer sur les résultats : produire des figures pour des rapports
  ou des publications, écrire des présentations.

Cahier des charges
---------------------

* Riche collections de **briques** déjà toutes programmées correspondant
  aux méthodes numériques classiques ou aux actions de base : on ne veut 
  pas avoir à reprogrammer le tracé d'une courbe, un algorithme de
  transformée de Fourier ou de fit. Minimiser la réinvention de la roue !

* Rapide à apprendre : l'informatique n'est pas notre métier ni notre
  formation. On veut pouvoir tracer une courbe, lisser un signal, faire
  une transformée de Fourier sans avoir besoin de chercher des heures 
  comment faire.

* Communication facile avec les collaborateurs, étudiants, clients, pour
  faire vivre le code au sein d'un labo ou d'une entreprise : le
  code devrait pouvoir se lire comme un livre. Le langage doit donc
  contenir le moins possible de syntaxe ou de procédures "parasite", 
  qui détournent l'attention de la signification
  mathématique/scientifique du code

* Codes efficaces qui s'exécutent rapidement... mais rien ne sert de faire 
  du code rapide si on met plus de temps à l'écrire qu'à l'exécuter. 
  Il faut donc une bonne vitesse de développement et une bonne vitesse 
  d'exécution.

* Un seul environnement/langage pour tout faire, si possible, pour ne pas
  réapprendre un logiciel à chaque nouveau problème.

Solutions existantes
----------------------

Quelles solutions les scientifiques utilisent-ils pour travailler ?

* Langages compilés : C, C++, Fortran, etc.

    * Avantages :

	* Très grande efficacité. Compilateurs très optimisés. Pour des
	  calculs "brute force", c'est difficile de battre la vitesse
	  d'éxécution des codes programmés avec ces langages. 

	* Des librairies scientifiques très optimisées ont été écrites
	  pour ces langages. Ex : blas (manipulation de tableaux de nombres).

    * Inconvénients :

	* Utilisation lourde : pas d'interactivité dans le développement
	  (étape de compilation peut être pénible). Syntaxe "verbeuse"
	  (&, ::, }}, ; etc.). Gestion de la mémoire délicate en C. Ce
	  sont des **langages difficiles** à manier pour le
	  non-informaticien.

* Langages de scripts : Matlab

    * Avantages : 

	* collection très riche de librairies avec de nombreux
          algorithmes, dans des domaines très variés. Exécution rapide
	  car les librairies sont souvent écrites dans un langage
	  compilé.

	* environnement de développement très agréable : aide complète et
	  bien organisée, éditeur intégré, etc.

	* support commercial disponible

    * Inconvénients : 

	* langage de base assez pauvre, qui peut se révéler limitant pour
          des utilisations avancées.

	* prix élevé

* Autres langages de scripts : Scilab, Octave, Igor, R, IDL, etc.

    * Avantages : libres/gratuits/moins chers que Matlab. Certaines
      fonctionnalités peuvent être très développées (statistiques dans
      R, figures pour Igor, etc.). 

    * Inconvénients : moins d'algorithmes disponibles que dans Matlab, et
      langage pas plus évolué. 

* Logiciels spécialisés pour une utilisation. Ex : Gnuplot ou xmgrace
  pour tracer des courbes. Ces logiciels sont très puissants, par contre
  leur utilisation est limitée au seul plot. C'est dommage d'avoir à
  apprendre un logiciel rien que pour ça.

* **Et Python ?**

    * Avantages :
    
	* Librairies très riches de calcul scientifique (un peu moins de
	  choses que dans Matlab cependant).
    
	* Langage très bien pensé permettant d'écrire du code très
	  lisible, et très bien structuré : on "code ce qu'on pense".

	* Librairies pour d'autres applications que le calcul
	  scientifique (gestion d'un serveur web, d'un port série, etc.). 

	* Logiciel libre, largement distribué avec une communauté
	  dynamique d'utilisateurs.

    * Inconvénients :  

	* environnement de développement un peu moins agréable que par
	  exemple Matlab (c'est un peu plus "pour les geeks").

	* les librairies scientifiques ne proposent pas l'intégralité des
	  toolboxes de Matlab (en contrepartie, certaines parties de ces
	  librairies peuvent être plus complètes ou mieux faites).


