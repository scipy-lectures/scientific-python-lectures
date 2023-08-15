.. _help:

Getting help and finding documentation
=========================================

**Author**: *Emmanuelle Gouillart*

Rather than knowing all functions in NumPy and SciPy, it is important to
find rapidly information throughout the documentation and the available
help. Here are some ways to get information:

* In Ipython, ``help function`` opens the docstring of the function. Only
  type the beginning of the function's name and use tab completion to
  display the matching functions.

  .. ipython::

      @verbatim
      In [204]: help(np.van<TAB>

      In [204]: help(np.vander)

In Ipython it is not possible to open a separated window for help and
documentation; however one can always open a second ``Ipython`` shell
just to display help and docstrings...

* Numpy's and Scipy's documentations can be browsed online on
  https://scipy.org and https://numpy.org. The ``search`` button is quite
  useful inside
  the reference documentation of the two packages.

  Tutorials on various topics as well as the complete API with all
  docstrings are found on this website.

* Numpy's and Scipy's documentation is enriched and updated on a regular
  basis by users on a wiki https://numpy.org/doc/stable/. As a result,
  some docstrings are clearer or more detailed on the wiki, and you may
  want to read directly the documentation on the wiki instead of the
  official documentation website. Note that anyone can create an account on
  the wiki and write better documentation; this is an easy way to
  contribute to an open-source project and improve the tools you are
  using!

* The SciPy Cookbook https://scipy-cookbook.readthedocs.io gives recipes on many
  common problems frequently encountered, such as fitting data points,
  solving ODE, etc.

* Matplotlib's website https://matplotlib.org/ features a very
  nice **gallery** with a large number of plots, each of them shows both
  the source code and the resulting plot. This is very useful for
  learning by example. More standard documentation is also available.


Finally, two more "technical" possibilities are useful as well:

* In Ipython, the magical function ``%psearch`` search for objects
  matching patterns. This is useful if, for example, one does not know
  the exact name  of a function.


  .. ipython::

      In [3]: import numpy as np
      In [4]: %psearch np.diag*

* numpy.lookfor looks for keywords inside the docstrings of specified modules.

  .. ipython::
      :okwarning:

      In [45]: np.lookfor('convolution')

* If everything listed above fails (and Google doesn't have the
  answer)... don't despair! There is a vibrant Scientific Python community.
  Scientific Python is present on various platform.
  https://scientific-python.org/community/


  Packages like SciPy and NumPy also have their own channels. Have a look at
  their respective websites to find out how to engage with users and
  maintainers.
