.. _guide

=================
How to contribute
=================

**Author**: *Nicolas Rougier*

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
and different parts may be combined into a course on Python for scientific
computing. Thus you want them to be quite interactive and reasonably short
(one to two hours) or your audience might just fall asleep long before you've
finished talking...

Last but not least, the goal of this material is to provide a concise text
useful to learning the main features of the scipy ecosystem. If you want to
contribute to reference material, we suggest that you contribute to the
documentation of the specific packages that you are interested in.

Keeping it concise: collapsing paragraphs
===========================================

The HTML output is used for displaying on screen while teaching. The goal
is to have the same material displayed as in the notes. Thus there needs
to be a very concise display, with bullet-lists rather than full-blown
paragraphs and sentences. However, in the long run, it is useful to have
more elaborate discussions that people can read and refer to. For this,
the ``tip`` sphinx directive will create collapsible paragraphs, that can
be hidden during an oral presentation::

    .. tip::

        Here insert a full-blown discussion, that will be collapsable in
        the HTML version.

        It can span on multiple paragraphs

This renders as following:

    .. tip::

        Here insert a full-blown discussion, that will be collapsable in
        the HTML version.

        It can span on multiple paragraphs

Figures and code examples
==========================

**We do not check figures in the repository**.
Any figure must be generated from a python script that needs to be named
``plot_xxx.py`` (xxx can be anything of course) and put into the ``examples``
directory. The generated image will be named from the script name.

.. image::  auto_examples/images/plot_simple_1.png
   :target: auto_examples/plot_simple.html


This is the way to include your image and link it to the code:

.. code-block:: rst

   .. image::  auto_examples/images/plot_simple_1.png
      :target: auto_examples/plot_simple.html

You can display the corresponding code using the ``literal-include``
directive.

.. literalinclude:: examples/plot_simple.py

.. note::

    The code to provide this style of plot inclusion was adopted from the
    scikits.learn project and can be found in ``sphinxext/gen_rst.py``.

Using Markup
============

There are three main kinds of markup that should be used: *italics*, **bold**
and ``fixed-font``. *Italics* should be used when introducing a new technical
term, **bold** should be used for emphasis and ``fixed-font`` for source code.

.. topic:: Example:

    When using *object-oriented programming* in Python you **must** use the
    ``class`` keyword to define your *classes*.

In restructured-text markup this is::

    when using *object-oriented programming* in Python you **must** use the
    ``class`` keyword to define your *classes*.


Linking to package documentations
==================================

The goal of the scipy lecture notes is not to duplicate or replace the
documentation of the various packages. You should link as much as
possible to the original documentation.

For cross-referencing API documentation we prefer to use the `intersphinx
extension <http://sphinx-doc.org/latest/ext/intersphinx.html>`_. This provides
the directives `:mod:`, `:class:` and `:func:` to cross-link to modules,
classes and functions respectively. For example the ``:func:`numpy.var``` will
create a link like :func:`numpy.var`.

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


Using Github
============

The easiest way to make your own version of this teaching material
is to fork it under Github, and use the git version control system to
maintain your own fork. For this, all you have to do is create an account
on github (this site) and click on the *fork* button, on the top right of this
page. You can use git to pull from your *fork*, and push back to it the
changes. If you want to contribute the changes back, just fill a
*pull request*, using the button on the top of your fork's page.

Please refrain from modifying the Makefile unless it is absolutely
necessary.

Admonitions
============

.. note:: 
   
   This is a note

.. warning::

   This is a warning

Clearing floats
================

Figures positionned with `:align: right` are float. To flush them, use::

    |clear-floats|

References
==========

.. target-notes::

.. _`Documentation style guide`: http://documentation-style-guide-sphinx.readthedocs.org/en/latest/style-guide.html
.. _`tips, tricks`: http://docness.readthedocs.org/en/latest/index.html
