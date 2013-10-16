========================
Interfacing with Fortran
========================

:author: Olav Vahtras

.. topic:: Foreword

This chapter contains basic interfacing with Fortran functions and subroutines. 

.. contents:: Chapters contents
   :local:
   :depth: 1

Introduction
============

Fortran is perhaps the easiest of the compiled languages to use for building python modules, with only minimal modifications in the original Fortran source, and from the python side there is no apparent difference from importing ordinary module routines.  As always, a combiled languages is a good candidate

* when python is too slow
* when an algorithm prevents vectorization

Python and Fortran functions
----------------------------
To illustrate the use of fortran functions we consider ``cos(x)``
In this special case we call the interface ``cos_fun`` not to hide the intrinsic

::

    #cos_module.F
    double precision function cos_fun(x)
    double precision x
    cos_fun = cos(x)
    return
    end


We now use the `f2py` command (provided with numpy) to generate a dynamically linked library python can import
::

    $ f2py -c -m cos_module cos_module.f > log

This generates the dynamically linked library ``cos_module.so``

In python we import and use the Fortran version as any other module function
::
    >>> import math
    >>> from cos_module import cos_fun as cos
    >>> print cos(math.pi/3)
    0.5


Fortran subroutines
-------------------
In fortran input and output arguments to fortran subroutines can be in any order.  The python convention is that arguments are input and return values are output.  We can supply the subroutine with information, compiler directives for ``f2py``
::
 
          subroutine cos_sub(cos_x, x)
          double precision cos_x, x
    Cf2py intent(out) cos_x
          cos_x = cos(x)
          return
          end

The default for ``f2py`` is to assume defualt input intent, so there is only one directive for the output variable.
The subroutine is called from python as follows
::

    >>> import math
    >>> from cos_module import cos_sub as cos
    >>> print cos(math.pi/3)
    0.5


Fortran and numpy arrays
========================

A subroutine with in/out intent arguments
-----------------------------------------

We now extend the ``cos_sub`` routine to accept a numpy array. We make the
interface such that we provide an array with x-values as input and an empty array
for the return values.  One reason for this is to avoid a potential source of
memory leaks in terms of new allocations for every function call. 
We thus create the array in python and declare it to have the in/out
intent attribute for ``f2py``. Note that the dimension of the array ``n`` the subroutine
definition, is not required in a call from python, as the size of the array is
a property of the array

::

          subroutine vec_cos_sub(cos_x, x, n)
          double precision cos_x, x
          dimension cos_x(n), x(n)
    Cf2py intent(in) x
    Cf2py intent(in, out) cos_x
          do i=1, n
             cos_x(i) = cos(x(i))
          end do
          return
          end

Calling this vectorized version is done with the following code
::

    >>> vec = numpy.array([0, math.pi/6, math.pi/3, math.pi/2])
    >>> from cos_module import vec_cos_sub as cos
    >>> cv = numpy.zeros(len(vec))
    >>> cv = cos(cv, vec)
    >>> print cv
    [  1.00000000e+00   8.66025404e-01   5.00000000e-01   6.12323400e-17]

