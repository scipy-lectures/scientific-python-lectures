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

Advantages
----------

* Requires no additional libraries
* Lot's of low-level control

Disadvantages
-------------

* Requires a substantial amount of effort
* Much overhead in the code
* Must be compiled

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
