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

A complete list of requirements with information on the versions presently used
to build the SciPy lecture notes, can be inferred from the
:download:`requirements file <requirements.txt>`:

.. literalinclude:: requirements.txt
    :start-after: # libraries before the special comment marked

The installation instructions below cover (i) Debian/Ubuntu with miniconda as this is the
method we use to test and build the material and (ii) generic instructions using pip.

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

.. topic:: **Installing the Python packages with pip**

    `pip <https://pip.pypa.io/>`_ is the recommended tool to install Python
    packages. However, a number of packages must be installed from the system. Install
    Python, the Python "headers" (the `python3-dev` package on Debian, for instance), a C
    development environment, `GNU Make <https://www.gnu.org/software/make/>`_, a full LaTeX
    distribution (e.g. `TeX Live <https://www.tug.org/texlive/>`_ with extra fonts and
    utilities) and `git <https://git-scm.com/>`_.

    The remaining software can be installed with `pip`. Depending on the configuration, the
    probably most convenient and safe way to proceed is with::

        pip install --user -r requirements.txt

    This will install the software in a local directory belonging to the user and will not
    interfere with system-wide Python packages.
