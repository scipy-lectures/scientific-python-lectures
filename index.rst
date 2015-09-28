Scipy Lecture Notes 
===========================

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
        font-family: FontAwesome, sans-serif;
        min-width: 200pt;
    }

    div.sidebar ul {
        list-style: none;
        text-indent: -3ex;
        color: #555;
    }

    div.sidebar li {
        margin-top: .5ex;
    }

    div.preface {
        margin-top: 20px;
    }

  </style>

.. nice layout in the toc

.. include:: tune_toc.rst 

.. |pdf| unicode:: U+f1c1 .. PDF file

.. |archive| unicode:: U+f187 .. archive file

.. |github| unicode:: U+f09b  .. github logo

.. only:: html

    .. sidebar::  Download 
       
       * |pdf| `PDF, 2 pages per side <./_downloads/ScipyLectures.pdf>`_

       * |pdf| `PDF, 1 page per side <./_downloads/ScipyLectures-simple.pdf>`_
   
       * |archive| `HTML and example files <https://github.com/scipy-lectures/scipy-lectures.github.com/zipball/master>`_
     
       * |github| `Source code (github) <https://github.com/scipy-lectures/scipy-lecture-notes>`_


    Tutorials on the scientific Python ecosystem: a quick introduction to
    central tools and techniques. The different chapters each correspond
    to a 1 to 2 hours course with increasing level of expertise, from
    beginner to expert.

    .. rst-class:: preface

        .. toctree::
            :maxdepth: 2

            preface.rst

|

.. rst-class:: tune

  .. toctree::
    :numbered:

    intro/index.rst
    advanced/index.rst
    packages/index.rst

|

..  
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.

.. only:: html

 .. raw:: html
 
   <div style='display: none; height=0px;'>

 :download:`ScipyLectures.pdf` :download:`ScipyLectures-simple.pdf`
 
 .. image:: themes/plusBox.png

 .. image:: images/logo.svg

 .. raw:: html
 
   </div>
   </small>


..
    >>> # For doctest on headless environments (needs to happen early)
    >>> import matplotlib
    >>> matplotlib.use('Agg')




