Python Scientific Lecture Notes 
===================================================================

.. only:: html

   One document to learn numerics, science, and data with Python
   --------------------------------------------------------------

.. raw html to center the title

.. raw:: html

  <style type="text/css">
    div.documentwrapper h1 {
        text-align: center;
        font-size: 240% ;
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

    h1:hover > a.headerlink,
    h2:hover > a.headerlink
    {
        visibility: hidden;
    }

    div.sidebar {
        margin-right: -20px;
        margin-top: -10px;
        border-radius: 6px;
    }

    div.preface {
        margin-top: 20px;
    }

  </style>

.. nice layout in the toc

.. include:: tune_toc.rst 

.. only:: html

    .. sidebar::  Download 
       
       * `PDF, 2 pages per side <./_downloads/PythonScientific.pdf>`_

       * `PDF, 1 page per side <./_downloads/PythonScientific-simple.pdf>`_
   
       * `HTML and example files <https://github.com/scipy-lectures/scipy-lectures.github.com/zipball/master>`_
     
       * `Source code (github) <https://github.com/scipy-lectures/scipy-lecture-notes>`_


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
____


.. only:: html

 .. raw:: html

   <small style="color: gray">

 Version: |version| (output of ``git describe`` for `project repository`_)

 .. _`project repository`: https://github.com/scipy-lectures/scipy-lecture-notes

..  
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.

.. only:: html

 .. raw:: html
 
   <div style='visibility: hidden ; height=0'>

 :download:`ScipyLectures.pdf` :download:`ScipyLectures-simple.pdf`
 
 .. image:: themes/plusBox.png

 .. raw:: html
 
   </div>
   </small>


..
    >>> # For doctest on headless environments (needs to happen early)
    >>> import matplotlib
    >>> matplotlib.use('Agg')




