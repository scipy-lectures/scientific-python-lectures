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

* `Python-C-Api <http://docs.python.org/2/c-api/>`_
* `Ctypes <http://docs.python.org/2/library/ctypes.html>`_
* `SWIG (Simplified Wrapper and Interface Generator) <http://www.swig.org/>`_
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

The `Python-C-API <http://docs.python.org/2/c-api/>`_ is the backbone of the
standard Python interpreter (a.k.a *CPython*). Using this API it is possible to
write Python extension module in C and C++. Obviously, these extension modules
can, by virtue of language compatibility, call any function written in C or
C++.

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

Now let's see how robust this is:

.. sourcecode:: ipython

    In [10]: cos_module.cos_func('foo')
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-10-11bee483665d> in <module>()
    ----> 1 cos_module.cos_func('foo')

    TypeError: a float is required


Ctypes
======

`Ctypes <http://docs.python.org/2/library/ctypes.html>`_ is a *foreign
function library* for Python. It provides C compatible data types, and allows
calling functions in DLLs or shared libraries. It can be used to wrap these
libraries in pure Python.

Advantages
----------

* Part of the python standard library
* Does not need to be compiled
* Wrapping code entirely in Python

Disadvantages
-------------

* Requires code to be wrapped to be available as a shared library
  (roughly speaking ``*.dll`` in Windows ``*.so`` in Linux and ``*.dylib`` in Mac OSX.)

Example
-------

As advertised, the wrapper code is in pure python.

.. literalinclude:: ctypes/cos_module.py
   :language: python

* Finding and loading the library may vary depending on your operating system,
  check the documentation for details.
* This may be somewhat deceptive, since the math library exists in compiled
  form on the system already. If you were to wrap a in-house library, you would
  have to compile it first, which may or may not require some additional effort.

We may now use this, as before:

.. sourcecode:: ipython

    In [1]: import cos_module

    In [2]: cos_module?
    Type:       module
    String Form:<module 'cos_module' from 'cos_module.py'>
    File:       /home/esc/git-working/scipy-lecture-notes/advanced/interfacing_with_c/ctypes/cos_module.py
    Docstring:  <no docstring>

    In [3]: dir(cos_module)
    Out[3]:
    ['__builtins__',
     '__doc__',
     '__file__',
     '__name__',
     '__package__',
     'cos_func',
     'ctypes',
     'find_library',
     'libm']

    In [4]: cos_module.cos_func(1.0)
    Out[4]: 0.5403023058681398

    In [5]: cos_module.cos_func(0.0)
    Out[5]: 1.0

    In [6]: cos_module.cos_func(3.14159265359)
    Out[6]: -1.0

As with the previous example, this code is somewhat robust, although the error
message is not quite as helpful, since it does not tell us what the type should be.

.. sourcecode:: ipython

    In [7]: cos_module.cos_func('foo')
    ---------------------------------------------------------------------------
    ArgumentError                             Traceback (most recent call last)
    <ipython-input-7-11bee483665d> in <module>()
    ----> 1 cos_module.cos_func('foo')

    /home/esc/git-working/scipy-lecture-notes/advanced/interfacing_with_c/ctypes/cos_module.py in cos_func(arg)
         12 def cos_func(arg):
         13     ''' Wrapper for cos from math.h '''
    ---> 14     return libm.cos(arg)

    ArgumentError: argument 1: <type 'exceptions.TypeError'>: wrong type

SWIG
====

`SWIG <http://www.swig.org/>`_, the Simplified Wrapper Interface Generator,
is a software development tool that connects programs written in C and C++
with a variety of high-level programming languages, including Python. The
important thing with SWIG is, that it can autogenerate the wrapper code for you.
While this is an advantage in terms of development time, it can also be a
burden. The generated file tend to be quite large and may not be too human
readable and the multiple levels of indirection which are a result of
the wrapping process, may be a bit tricky to understand.

Advantages
----------

* Can automatically wrap entire libraries given the headers
* Works nicely with C++

Disadvantages
-------------

* Autogenerates enormous files
* Hard to debug if something goes wrong

Example
-------

Let's imagine that our ``cos`` function lives in a ``cos_module`` which has
been written in ``c`` and consists of the source file ``cos_module.c``:

.. literalinclude:: swig/cos_module.c
    :language: c

and the header file ``cos_module.h``:

.. literalinclude:: swig/cos_module.h
    :language: c

And our goal is to expose the ``cos_func`` to Python. To achieve this with
SWIG, we must write an *interface file* which contains the instructions for SWIG.

.. literalinclude:: swig/cos_module.i
    :language: c

As you can see, not too much code is needed here. For this simple example it is
enough to simply include the header file in the interface file, to expose the
function to Python. However, SWIG does allow for more fine grained
inclusion/exclusion of functions found in header files, check the documentation
for details.

Generating the compiled wrappers is a two stage process:

#. Run the ``swig``  executable on the interface file to generate the files
   ``cos_module_wrap.c``, which is the source file for the autogenerated python c-extension
   and ``cos_module.py``, which is the autogenerated pure python module.

#. Compile the ``cos_module_wrap.c`` into the ``_cos_module.so``. Luckily,
   ``distutils`` knows how to handle SWIG interface files, so that our
   ``setup.py`` is simply:

.. literalinclude:: swig/setup.py
    :language: python

.. sourcecode:: console

    $ cd advanced/interfacing_with_c/swig

    $ ls
    cos_module.c  cos_module.h  cos_module.i  setup.py

    $ python setup.py build_ext --inplace
    running build_ext
    building '_cos_module' extension
    swigging cos_module.i to cos_module_wrap.c
    swig -python -o cos_module_wrap.c cos_module.i
    creating build
    creating build/temp.linux-x86_64-2.7
    gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/esc/anaconda/include/python2.7 -c cos_module.c -o build/temp.linux-x86_64-2.7/cos_module.o
    gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/esc/anaconda/include/python2.7 -c cos_module_wrap.c -o build/temp.linux-x86_64-2.7/cos_module_wrap.o
    gcc -pthread -shared build/temp.linux-x86_64-2.7/cos_module.o build/temp.linux-x86_64-2.7/cos_module_wrap.o -L/home/esc/anaconda/lib -lpython2.7 -o /home/esc/git-working/scipy-lecture-notes/advanced/interfacing_with_c/swig/_cos_module.so

    $ ls
    build/  cos_module.c  cos_module.h  cos_module.i  cos_module.py  _cos_module.so*  cos_module_wrap.c  setup.py

We can now load and execute the ``cos_module`` as we have done in the previous examples:

.. sourcecode:: ipython

    In [1]: import cos_module

    In [2]: cos_module?
    Type:       module
    String Form:<module 'cos_module' from 'cos_module.py'>
    File:       /home/esc/git-working/scipy-lecture-notes/advanced/interfacing_with_c/swig/cos_module.py
    Docstring:  <no docstring>

    In [3]: dir(cos_module)
    Out[3]:
    ['__builtins__',
     '__doc__',
     '__file__',
     '__name__',
     '__package__',
     '_cos_module',
     '_newclass',
     '_object',
     '_swig_getattr',
     '_swig_property',
     '_swig_repr',
     '_swig_setattr',
     '_swig_setattr_nondynamic',
     'cos_func']

    In [4]: cos_module.cos_func(1.0)
    Out[4]: 0.5403023058681398

    In [5]: cos_module.cos_func(0.0)
    Out[5]: 1.0

    In [6]: cos_module.cos_func(3.14159265359)
    Out[6]: -1.0

Again we test for robustness:

.. sourcecode:: ipython

    In [7]: cos_module.cos_func('foo')
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-7-11bee483665d> in <module>()
    ----> 1 cos_module.cos_func('foo')

    TypeError: in method 'cos_func', argument 1 of type 'double'

Cython
======

Advantages
----------

Disadvantages
-------------
