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

Editorial policy
..................

The goal of this material is to provide a concise text useful to
learning the main features of the scipy ecosystem. If you want
to contribute to reference material, we suggest that you contribute
to the documentation of the specific packages that you are
interested in.

The HTML output can be used for displaying on screen while
teaching. The goal is to have the same material displayed as
in the notes. This is why the HTML version should be kept concise, with
bullet-lists rather than full-blown paragraphs and sentences.
In the long run, we would like to build more elaborate discussions. For this,
the policy is to use the::

   .. only:: pdf

sphinx directive.

Each chapter should be kept reasonably short: 1 to 2 hours of tutorial.
The reason is two-fold. First these chapters are atoms that can be
combined to build a course on scientific computing with Python. Second,
people's attention span does not go much beyond an hour or two, whether
they are reading a tutorial or following it in a class room.

Using Markup
............

There are three main kinds of markup that should be used: *italics*, **bold**
and ``fixed-font``. *Italics* should be used when introducing a new technical
term, **bold** should be used for emphasis and ``fixed-font`` for source code.

.. topic:: Example:

    When using *object-oriented programming* in Python you **must** use the
    ``class`` keyword to define your *classes*.

In restructured-text markup this is::

    when using *object-oriented programming* in Python you **must** use the
    ``class`` keyword to define your *classes*.

Modifying
.........

The easiest way to make your own version of this teaching material
is to fork it under Github, and use the git version control system to
maintain your own fork. For this, all you have to do is create an account
on github (this site) and click on the *fork* button, on the top right of this
page. You can use git to pull from your *fork*, and push back to it the
changes. If you want to contribute the changes back, just fill a
*pull request*, using the button on the top of your fork's page.

Please refrain from modifying the Makefile unless it is absolutely
necessary.

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
