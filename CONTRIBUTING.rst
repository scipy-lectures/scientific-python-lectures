Contributing
=============

The Scientific Python Lectures are a community-based effort and require
constant maintenance and improvements. New contributions such as wording
improvements or inclusion of new topics are welcome.

To propose bugfixes or straightforward improvements to the lectures, see the
contribution guide below.

For new topics, read the objectives first and `open an issue on the GitHub
project <https://github.com/scipy-lectures/scientific-python-lectures/issues>`_ to
discuss it with the editors.


Objectives and design choices for the lectures
----------------------------------------------

Contributors should keep the following objectives and design choices of
the Scientific Python Lectures in mind.

Objectives:

* Provide a self-contained introduction to Python and its primary computational
  packages, the ”Scientific Python stack“.
* Provide tutorials for a selection of widely-used and stable computational
  libraries.
  Currently, we cover pandas, statmodels, seaborn, scikit-image,
  scikit-learn, and sympy.
* Automated testing is applied to the code examples as much as possible.

Design choices:

* Each chapter should provide a useful basis for a 1‒2 h tutorial.
* The code should be readable.
* An idomatic style should be followed, e.g. ``import numpy as np``,
  preference for array operations, PEP8 coding conventions.


Contributing guide
------------------

The directory ``guide`` contains instructions on how to contribute:

.. topic::  **Example chapter**

  .. toctree::

   guide/index.rst

Building instructions
----------------------

To generate the html output for on-screen display, Type::

    make html

the generated html files can be found in ``build/html``

The first build takes a long time, but information is cached and
subsequent builds will be faster.

To generate the pdf file for printing::

    make pdf

The pdf builder is a bit difficult and you might have some TeX errors.
Tweaking the layout in the ``*.rst`` files is usually enough to work
around these problems.

Requirements
............

Build requirements are listed in the
:download:`requirements file <requirements.txt>`:

.. literalinclude:: requirements.txt

Ensure that you have a `virtual environment
<https://docs.python.org/3/library/venv.html>`__ or conda environment
set up, then install requirements with::

  pip install -r requirements.txt

Note that you will also need the following system packages:

 - Python C development headers (the `python3-dev` package on Debian, e.g.),
 - a C compiler like gcc,
 - `GNU Make <https://www.gnu.org/software/make/>`__,
 - a full LaTeX distribution such as `TeX Live
   <https://www.tug.org/texlive/>`__ (``texlive-latex-base``,
   ``texlive-latex-extra``, ``texlive-fonts-extra``, and ``latexmk``
   on Debian/Ubuntu),
 - `dvipng <http://savannah.nongnu.org/projects/dvipng/>`__,
 - `latexmk <https://personal.psu.edu/~jcc8/software/latexmk/>`__,
 - `git <https://git-scm.com/>`__.

Updating the cover
..................

Use inkscape to modify the cover in ``images/``, then export to PDF::

  inkscape --export-filename=cover-2024.pdf cover-2024.svg

Ensure that the ``images/cover.pdf`` symlink points to the correct
file.
