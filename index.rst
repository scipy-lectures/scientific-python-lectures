Scientific Python Lectures
==========================

.. only:: html

   One document to learn numerics, science, and data with Python
   --------------------------------------------------------------

.. raw html to center the title

.. raw:: html

  <style type="text/css">
    div.documentwrapper h1 {
        text-align: center;
        font-size: 280% ;
        font-weight: bold;
        margin-bottom: 4px;
    }

    div.documentwrapper h2 {
        background-color: white;
        border: none;
        font-size: 130%;
        text-align: center;
        margin-bottom: 40px;
        margin-top: 4px;
    }

    a.headerlink:after {
        content: "";
    }

    div.sidebar {
        margin-right: -20px;
        margin-top: -10px;
        border-radius: 6px;
        font-family: sans-serif;
        min-width: 200pt;
    }

    div.sidebar ul {
        list-style: none;
        text-indent: -3ex;
        color: #555;
    }

    @media only screen and (max-width: 1080px) and (-webkit-min-device-pixel-ratio: 2), (max-width: 70ex)  {
        div.sidebar ul {
            text-indent: 0ex;
        }
    }

    div.sidebar li {
        margin-top: .5ex;
    }

    div.preface {
        margin-top: 20px;
    }

    .vcenter {
      vertical-align: sub;
    }

  </style>

.. nice layout in the toc

.. Icons from https://fonts.google.com/icons

.. |pdf-icon| image:: images/icon-pdf.svg
   :width: 15em
   :class: vcenter
   :alt: PDF icon

.. |html-icon| image:: images/icon-archive.svg
   :width: 15em
   :class: vcenter
   :alt: Archive icon


.. |github-icon| image:: images/icon-github.svg
   :width: 15em
   :class: vcenter
   :alt: GitHub icon


.. only:: html

    .. sidebar:: Download

       |pdf-icon| `PDF, 2 pages per side <./_downloads/ScientificPythonLectures.pdf>`_

       |pdf-icon| `PDF, 1 page per side <./_downloads/ScientificPythonLectures-simple.pdf>`_

       |github-icon| `Source code (github) <https://github.com/scipy-lectures/scientific-python-lectures>`_


    Tutorials on the scientific Python ecosystem: a quick introduction to
    central tools and techniques. The different chapters each correspond
    to a 1 to 2 hours course with increasing level of expertise, from
    beginner to expert.

    Release: |release|

    .. rst-class:: preface

        .. toctree::
            :maxdepth: 2

            preface.rst

|

.. rst-class:: tune

  .. toctree::
    :numbered: 4

    intro/index.rst
    advanced/index.rst
    packages/index.rst
    about.rst

|

..
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.

.. only:: html

 .. raw:: html

   <div style='display: none; height=0px;'>

 :download:`ScientificPythonLectures.pdf` :download:`ScientificPythonLectures-simple.pdf`

 .. image:: themes/plusBox.png

 .. image:: images/logo.svg

 .. raw:: html

   </div>
   </small>


..
    >>> # For doctest on headless environments (needs to happen early)
    >>> import matplotlib
    >>> matplotlib.use('Agg')
