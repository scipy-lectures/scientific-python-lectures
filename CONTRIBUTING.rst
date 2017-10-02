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

* make
* sphinx (>= 1.0)
* pdflatex
* pdfjam
* matplotlib
* scikit-learn (>= 0.8)
* scikit-image
* pandas
* seaborn

For a complete list of requirements with information on the versions presently used
to build the SciPy lecture notes, see ``requirements.txt``.

|
|

.. topic:: **Building on Debian/Ubuntu**

    The environment needed to create an html version of the SciPy lecture notes
    can be based on miniconda. We first download the latest version of miniconda
    and rename it to ``miniconda.sh`` for simplicity::

       wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    To ensure that the file has been downloaded correctly, one can compare the
    MD5 sum obtained by means of::

        md5sum miniconda.sh

    with the value listed on https://repo.continuum.io/miniconda/ . Miniconda can
    now be installed::

        bash miniconda.sh
        
    Review the license agreement and choose a target directory (here we assume it
    to be ``$HOME/miniconda3``). Unless you intend to work more extensively with
    miniconda, you do not want to modify ``.bashrc``. In this case, you need::

        export PATH=$HOME/miniconda3/bin:$PATH

    to find the correct binaries. Note that the path depends on the target directory
    chosen above.

    Now, we use ``environment.yml`` from the main directory of the ``scipy-lecture-notes``
    repository to install the required dependencies. Note that ``environment.yml`` yields
    an environment named ``testenv``. If you prefer a more telling name, make a copy of
    ``environment.yml``, replace ``testenv`` in line 4 by a more appropriate name and use
    this name in the following instead of ``testenv``. The conda environment is created
    by::

        conda env create -f environment.yml

    Now, the environment can be activated::

        source activate testenv

    and deactivated::

        source deactivate

    With an activated environment, you are now able to produce the html version of the
    SciPy as explained above. Generating a pdf version requires the system packages
    ``texlive``, ``texlive-latex-extra``, ``texlive-fonts-extra``, and ``latexmk``.

.. topic:: **Building on Fedora**

    As root::

        yum install python make python-matplotlib texlive-pdfjam texlive scipy \ 
        texlive-framed texlive-threeparttable texlive-wrapfig texlive-multirow
        pip install Sphinx
        pip install Cython
        pip install scikit-learn
        pip install scikit-image
