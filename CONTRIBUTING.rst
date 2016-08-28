Contributing
=============

The SciPy lecture notes are a community-based effort and require constant
maintenance and improvements. New contributions such as wording
improvements or inclusion of new topics are welcome.

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

*probably incomplete*

* make
* sphinx (>= 1.0)
* pdflatex
* pdfjam
* matplotlib
* scikit-learn (>= 0.8)
* scikit-image
* pandas
* seaborn

|
|

.. topic:: Building on Fedora

    As root::

        yum install python make python-matplotlib texlive-pdfjam texlive scipy \ 
        texlive-framed texlive-threeparttable texlive-wrapfig texlive-multirow
        pip install Sphinx
        pip install Cython
        pip install scikit-learn
        pip install scikit-image

