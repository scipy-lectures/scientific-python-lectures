.. _help:

Getting help and finding documentation
=========================================

.. only:: latex

    :author: Emmanuelle Gouillart

Rather than knowing all functions in Numpy and Scipy, it is important to
find rapidly information throughout the documentation and the available
help. Here are some ways to get information:

* In Ipython, ``help function`` opens the docstring of the function. Only
  type the beginning of the function's name and use tab completion to
  display the matching functions.

  .. sourcecode:: ipython
  
      In [204]: help np.v
      np.vander     np.vdot       np.version    np.void0      np.vstack
      np.var        np.vectorize  np.void       np.vsplit     
      
      In [204]: help np.vander
	
In Ipython it is not possible to open a separated window for help and
documentation; however one can always open a second ``Ipython`` shell
just to display help and docstrings...

* Numpy's and Scipy's documentations can be browsed online on
  http://docs.scipy.org/doc. The ``search`` button is quite useful inside
  the reference documentation of the two packages
  (http://docs.scipy.org/doc/numpy/reference/ and
  http://docs.scipy.org/doc/scipy/reference/). 

  Tutorials on various topics as well as the complete API with all
  docstrings are found on this website.


  .. image:: scipy_doc.png
     :align: center
     :scale: 80

* Numpy's and Scipy's documentation is enriched and updated on a regular
  basis by users on a wiki http://docs.scipy.org/numpy/. As a result,
  some docstrings are clearer or more detailed on the wiki, and you may
  want to read directly the documentation on the wiki instead of the
  official documentation website. Note that anyone can create an account on
  the wiki and write better documentation; this is an easy way to
  contribute to an open-source project and improve the tools you are
  using!

  .. image:: docwiki.png
     :align: center
     :scale: 80

* Scipy's cookbook http://www.scipy.org/Cookbook gives recipes on many
  common problems frequently encountered, such as fitting data points,
  solving ODE, etc. 


* Matplotlib's website http://matplotlib.sourceforge.net/ features a very
  nice **gallery** with a large number of plots, each of them shows both
  the source code and the resulting plot. This is very useful for
  learning by example. More standard documentation is also available. 


  .. image:: matplotlib.png
     :align: center
     :scale: 80

* Mayavi's website
  http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/
  also has a very nice gallery of examples
  http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/auto/examples.html
  in which one can browse for different visualization solutions.

  .. image:: mayavi_website.png
     :align: center
     :scale: 80

Finally, two more "technical" possibilities are useful as well:

* In Ipython, the magical function ``%psearch`` search for objects
  matching patterns. This is useful if, for example, one does not know
  the exact name  of a function.


  .. sourcecode:: ipython
  
      In [3]: import numpy as np
      In [4]: %psearch np.diag*
      np.diag
      np.diagflat
      np.diagonal

* numpy.lookfor looks for keywords inside the docstrings of specified modules.

  .. sourcecode:: ipython
  
      In [45]: numpy.lookfor('convolution')
      Search results for 'convolution'
      --------------------------------
      numpy.convolve
          Returns the discrete, linear convolution of two one-dimensional
      sequences.
      numpy.bartlett
          Return the Bartlett window.
      numpy.correlate
          Discrete, linear correlation of two 1-dimensional sequences.
      In [46]: numpy.lookfor('remove', module='os')
      Search results for 'remove'
      ---------------------------
      os.remove
          remove(path)
      os.removedirs
          removedirs(path)
      os.rmdir
          rmdir(path)
      os.unlink
          unlink(path)
      os.walk
          Directory tree generator.



* If everything listed above fails (and Google doesn't have the
  answer)... don't despair! Write to the mailing-list suited to your
  problem: you should have a quick answer if you describe your problem
  well. Experts on scientific python often give very enlightening
  explanations on the mailing-list.

    * **Numpy discussion** (numpy-discussion@scipy.org): all about numpy
      arrays, manipulating them, indexation questions, etc.


    * **SciPy Users List** (scipy-user@scipy.org): scientific computing
      with Python, high-level data processing, in particular with the
      scipy package.

    * matplotlib-users@lists.sourceforge.net for plotting with
      matplotlib.                               
                                             
