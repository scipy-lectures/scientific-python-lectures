==================
Interfacing with C
==================

:author: Valentin Haenel

.. topic:: Foreword

This chapter contains an *introduction* to the many different routes for making
your native code (primarliy ``C/C++``) available from python, a process
commonly referred to *wrapping*. The goal of this chapter is to give you a
flavour of what technologies exist and what their respective merits and
shortcomings are, so that you can select the appropriate one for your specific
needs. In any case, once you do start wrapping, you almost certainly will want
to consult the respective documentation for your selected technique.

.. contents:: Chapters contents
   :local:
   :depth: 1

Introduction
============

This chapter covers the following techniques:

* Python-C-Api
* Ctypes
* SWIG (Simplified Wrapper and Interface Generator)
* Cython

Having said that, there are other alternatives out there, but having understood
the basics of the ones above, you will be in a position to evaluate the
technique of your choice to see if it fits your needs.

Evaluation items

* Does it need to be compiled
* Does it play nicely with numpy

Warnings

* All of these techniques may crash the python interpreter. You have been
  warned.
* All the examples have been done on Linux

Objective -- what is your use-case

* Existing code in C/C++ that needs to be leveraged
* Python code too slow, bring inner loops to native code

The layered approach

...

Example
-------

A function from math.h, say cos, applied to some doubles.

Python-C-Api
============

The Python-C-API is the backbone of the standard Python interpreter (a.k.a
*CPython*). Using this API it is possible to write Python extension module in C
and C++. Obviously, these extension modules can, by virtue of language
compatibility, call any function written in C or C++.

When using the Python-C-API, one usually writes much boilerplate code, first to
parse the arguments that were given to a function, and later to construct the
return type.

Advantages
----------

* Requires no additional libraries
* Lot's of low-level control

Disadvantages
-------------

* Requires a substantial amount of effort
* Much overhead in the code
* Must be compiled

Example
-------

The following extension module, make the ``cos`` function from the standard
math library available to Python:

.. literalinclude:: python_c_api/cos_module.c
   :language: c

As you can see, there is much boilerplate, both to «massage» the arguments and
return types into place and for the module initialisation. Although some of
this is amortised, as the extension grows, the boilerplate required for each
function(s) remains.

The standard python build system ``distutils`` supports compiling Python-C-API
extensions from a ``setup.py``, which is rather convenient:

.. literalinclude:: python_c_api/setup.py
   :language: python

This can be compiled:

.. sourcecode:: console

    $ cd advanced/interfacing_with_c/python_c_api
    $ ls
    cos_module.c  setup.py
    $ python setup.py build_ext --inplace
    running build_ext
    building 'cos_module' extension
    creating build
    creating build/temp.linux-x86_64-2.7
    gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/esc/anaconda/include/python2.7 -c cos_module.c -o build/temp.linux-x86_64-2.7/cos_module.o
    gcc -pthread -shared build/temp.linux-x86_64-2.7/cos_module.o -L/home/esc/anaconda/lib -lpython2.7 -o /home/esc/git-working/scipy-lecture-notes/advanced/interfacing_with_c/python_c_api/cos_module.so
    $ ls
    build/  cos_module.c  cos_module.so  setup.py

* ``build_ext`` is to build extension modules
* ``--inplace`` will output the compiled extension module into the current directory

The file ``cos_module.so`` contains the compiled extension, which we can now load in the IPython interpreter:

.. sourcecode:: ipython

    In [1]: import cos_module

    In [2]: cos_module?
    Type:       module
    String Form:<module 'cos_module' from 'cos_module.so'>
    File:       /home/esc/git-working/scipy-lecture-notes/advanced/interfacing_with_c/python_c_api/cos_module.so
    Docstring:  <no docstring>

    In [3]: dir(cos_module)
    Out[3]: ['__doc__', '__file__', '__name__', '__package__', 'cos_func']

    In [4]: cos_module.cos_func(1.0)
    Out[4]: 0.5403023058681398

    In [5]: cos_module.cos_func(0.0)
    Out[5]: 1.0

    In [6]: cos_module.cos_func(3.14159265359)
    Out[7]: -1.0


Ctypes
======

Advantages
----------

* Does not need to be compiled, wrapping code entirely in Python

Disadvantages
-------------

* Requires code to be wrapped to be available as a shared library
  (roughly speaking ``*.ddl`` in Windows ``*.so`` in Linux and ``*.dylib`` in Mac OSX.)

Advantages
----------

SWIG
====

Advantages
----------

* Can automatically wrap entire libraries given the headers
* Works nicely with C++

Disadvantages
-------------

* Autogenerates enormous files
* Hard to debug if something goes wrong

Cython
======

Advantages
----------

Disadvantages
-------------
