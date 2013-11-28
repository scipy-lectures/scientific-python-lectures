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

Fortran is perhaps the easiest of the compiled languages to use for building
Python modules, with only minimal modifications in the original Fortran source.
In addition, from the Python side there is no apparent difference from importing
ordinary module functions.  As always, a compiled languages is a good candidate

* when Python is too slow
* when an algorithm prevents vectorization

Python and Fortran functions
----------------------------
To illustrate the use of Fortran functions we consider the cosine function``cos(x)``.
In this special case we call the interface function ``cos_fun`` so we do not hide the Fortran intrinsic.

::

    #cos_module.F
    double precision function cos_fun(x)
    double precision x
    cos_fun = cos(x)
    return
    end

We now use the `f2py` command (provided with numpy) to generate a dynamically
linked library which Python can import ::

    $ f2py -c -m cos_module cos_module.f

This generates the file ``cos_module.so``. In Python we import and use it as
with any other module ::

    >>> from cos_module import cos_fun as cos
    ... print cos(math.pi/3)
    0.5


Fortran subroutines
-------------------
In Fortran, input and output arguments to subroutines can be in any
order, whereas the Python convention is that function arguments are input and return values are
output.  We can supply the subroutine with information on input and output intent, as compiler directives for
``f2py``. The module functions that are generated translate the output variables to function return values
 ::
 
          subroutine cos_sub(cos_x, x)
          double precision cos_x, x
    Cf2py intent(out) cos_x
          cos_x = cos(x)
          return
          end

The default for ``f2py`` is to assume default input intent, so there is only one directive for the output variable.
This subroutine may be  called from Python as follows
::

    >>> from cos_module import cos_sub as cos
    >>> print cos(math.pi/3)
    0.5


Fortran and numpy arrays
========================

A subroutine interface 
----------------------

We now extend the ``cos_sub`` routine to accept a numpy array as an argument.
The length ``n`` of the array in the subroutine definition, is not required in
a call from Python, as the size of the array is a property of the array

::

          subroutine vec_cos_sub(cos_x, x, n)
          double precision cos_x, x
          dimension cos_x(n), x(n)
    Cf2py intent(in) x
    Cf2py intent(out) cos_x
          do i=1, n
             cos_x(i) = cos(x(i))
          end do
          return
          end

Declaring an array with output intent signals to ``f2py`` that it is to be
allocated. This subroutine is used from Python as follows, ::

    >>> vec = numpy.array([0, math.pi/6, math.pi/3, math.pi/2])
    >>> from cos_module import vec_cos_sub as cos
    >>> cv = cos(vec)
    >>> print cv
    [  1.00000000e+00   8.66025404e-01   5.00000000e-01   6.12323400e-17]

A subroutine with in/out intent arguments
-----------------------------------------

In some applications we may be interested in updating an existing array, so we
now make the interface where we provide an array with x-values as input and
an empty array for the return values.  Another reason for this approach is to avoid a
potential source of memory leaks that comes with new allocations for every
function call.  We thus create the array in Python and declare it to have the
in/out intent attribute for ``f2py``. 

::

          subroutine update_vec_cos_sub(cos_x, x, n)
          double precision cos_x, x
          dimension cos_x(n), x(n)
    Cf2py intent(in) x
    Cf2py intent(in, out) cos_x
          do i=1, n
             cos_x(i) = cos_x(i) + cos(x(i))
          end do
          return
          end

Calling this vectorized version is done with the following code
::

    >>> vec = numpy.array([0, math.pi/6, math.pi/3, math.pi/2])
    >>> from cos_module import update_vec_cos_sub as upd_cos
    >>> cv = numpy.zeros(len(vec))
    >>> cv = upd_cos(cv, vec)
    >>> print cv
    [  1.00000000e+00   8.66025404e-01   5.00000000e-01   6.12323400e-17]


For the interested reader a more detailed account of using Fortran with Python can be found in Langtangen: *Python Scripting for Computational Science* 

..
