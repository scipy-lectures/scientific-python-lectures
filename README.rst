This repository gathers some lecture notes on the scientific Python
ecosystem that can be used for a full course of scientific computing with
Python.

These documents are written with the rest markup language (.rst
extension) and built using Sphinx: http://sphinx.pocoo.org/.

Modifying
==========

Reusing and distributing
-------------------------

As stated in the LICENSE.txt file, this material comes with no strings
attached. Feel free to reuse and modify for your own teaching purposes.

However, we would like this reference material to be improved over time,
thus we encourage people to contribute back changes. These will be
reviewed and edited by the original authors.

Building instructions
----------------------

To generate the html output for on-screen display, Type::

    make html

the generated html files can be found in ``build/html``

The first build takes a long time, but information is cached and
subsequent builds will be faster.

To generate the pdf file for printing::

    make pdf

The pdf builder is a bit pointy and you might have some TeX errors. Tweaking
the layout in the rst files is usually enough to work around these
problems.

Requirements
............

*probably incomplete*

* make
* sphinx (>= 1.0)
* pdflatex
* pdfjam
* matplotlib
* scikit-learn (>= 0.8)

Contributing
-------------


Figures and code examples
..........................

The figure should be generated from Python source files. The policy is
to create an ``examples`` directory, in which you put the corresponding
Python files. Any files with a name starting with ``plot_`` will be run
during the build process, and figures created by matplotlib will be saved
as images in an ``auto_examples`` directory. You can use these to include
in the document as figures. To display the code snippet, you can use the
``literal-include`` directive. Any additional data needed by the plotting script
should be included in the same directory. NB: the code to provide this style of
plot inclusion was adopted from the scikits.learn project and can be found in
``sphinxext/gen_rst.py``.

.. topic:: Contributing guide and chapter example

   The directory `guide` contains an example chapter with specific
   instructions on how to contribute:

   .. toctree::

      guide/index.rst
