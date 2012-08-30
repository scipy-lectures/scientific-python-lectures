.. _guide

=================
How to contribute
=================

:authors: Nicolas Rougier

.. topic:: Foreword

   Use the ``topic`` keyword for any forewords


.. contents:: Chapters contents
   :local:
   :depth: 1


Make sure to read this `Documentation style guide`_ as well as these
`tips, tricks`_ and conventions about documentation content and workflows.


How to contribute ?
===================

Choose a topic that is not yet covered and write it up !

Create a new directory in either the ``intro`` or ``advanded`` part and start
writing into ``index.rst``. Don't forget to also update the ``index.rst`` in
the corresponding part such that your new tutorial appear in the table of
contents.

Also keep in mind that these tutorials are to being taught at different places
and you want them to be quite interactive or your audience might just fall
asleep long before you've finished talking...


Chapter, section, subsection, paragraph
=======================================

Try to avoid to go below paragraph granularity or your document might become
difficult to read:

.. code-block:: rst

   =============
   Chapter title
   =============

   Sample content.

   Section
   =======

   Subsection
   ----------

   Paragraph
   .........

   And some text.



Figures
=======

Any figure must be generated from a python script that needs to be named
``plot_xxx.py`` (xxx can be anything of course) and put into the ``examples``
directory. The generated image will be named from the script name.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   X = np.linspace(-np.pi,np.pi,100)
   Y = np.sin(X)
   plt.plot(X, Y, linewidth=2)
   plt.show()

.. image::  auto_examples/images/plot_simple.png
   :target: auto_examples/plot_simple.html


This is the way to include your image and link it to the code:

.. code-block:: rst

   .. image::  auto_examples/images/plot_simple.png
      :target: auto_examples/plot_simple.html


Admonitions
============

.. note:: 
   
   This is a note

.. warning::

   This is a warning

References
==========

.. target-notes::

.. _`Documentation style guide`: http://documentation-style-guide-sphinx.readthedocs.org/en/latest/style-guide.html
.. _`tips, tricks`: http://docness.readthedocs.org/en/latest/index.html
