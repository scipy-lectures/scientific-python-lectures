.. _tutorial:


.. TODO: bench and fit in 1:30

======================================
Sympy : Symbolic Mathematics in Python
======================================

.. role:: input(strong)

Introduction
============

SymPy is a Python library for symbolic mathematics. It aims become a
full featured computer algebra system thatn can compete directly with
commercial alternatives (Mathematica, Maple) while keeping the code as
simple as possible in order to be comprehensible and easily
extensible. Commercial alternatives include Mathematica and Maple.
SymPy is written entirely in Python and does not require any external
libraries.



First Steps with SymPy
======================


Using SymPy as a calculator
---------------------------

Sympy has three built-in numeric types: Real, Rational and Integer.

The Rational class represents a rational number as a pair of two Integers: the numerator and the denominator, so Rational(1,2) represents 1/2, Rational(5,2) 5/2 and so on.

::

    >>> from sympy import *
    >>> a = Rational(1,2)

    >>> a
    1/2

    >>> a*2
    1

    >>> Rational(2)**50/Rational(10)**50
    1/88817841970012523233890533447265625


SymPy uses mpmath in  the background, which makes it possible to perform computations using arbitrary - precission arithmetic. That way, some special constants, like e, pi, oo (Infinity), that are treated as symbols and
have arbitrary precission::

    >>> pi**2
    pi**2

    >>> pi.evalf()
    3.14159265358979

    >>> (pi+exp(1)).evalf()
    5.85987448204884

as you see, evalf evaluates the expression to a floating-point number

There is also a class representing mathematical infinity, called ``oo``::

    >>> oo > 99999
    True
    >>> oo + 1
    oo

Symbols
-------

In contrast to other Computer Algebra Systems, in SymPy you have to declare
symbolic variables explicitly::

    >>> from sympy import *
    >>> x = Symbol('x')
    >>> y = Symbol('y')

Then you can play with them::

    >>> x+y+x-y
    2*x

    >>> (x+y)**2
    (x + y)**2

    >>> ((x+y)**2).expand()
    2*x*y + x**2 + y**2

And substitute them for other symbols or numbers using ``subs(old, new)``::

    >>> ((x+y)**2).subs(x, 1)
    (1 + y)**2

    >>> ((x+y)**2).subs(x, y)
    4*y**2


Exercises
---------

  1. Calculate :math:`\sqrt{2}` with 100 decimals.
  2. Calculate :math:`\pi + 1` with 100 decimals.
  3. Calculate :math:`1/2 + 1/3` in rational arithmetic (without
    converting to floating point numbers).


Algebra
=======

One of the most cumbersome algebraic operations are partial fraction
decomposition.  For partial fraction decomposition, use ``apart(expr,
x)``::

    In [1]: 1/( (x+2)*(x+1) )
    Out[1]:
           1
    ───────────────
    (2 + x)*(1 + x)

    In [2]: apart(1/( (x+2)*(x+1) ), x)
    Out[2]:
      1       1
    ───── - ─────
    1 + x   2 + x

    In [3]: (x+1)/(x-1)
    Out[3]:
    -(1 + x)
    ────────
     1 - x

    In [4]: apart((x+1)/(x-1), x)
    Out[4]:
          2
    1 - ─────
        1 - x

To combine things back together, use ``together(expr, x)``::

    In [7]: together(1/x + 1/y + 1/z)
    Out[7]:
    x*y + x*z + y*z
    ───────────────
         x*y*z

    In [8]: together(apart((x+1)/(x-1), x), x)
    Out[8]:
    -1 - x
    ──────
    1 - x

    In [9]: together(apart(1/( (x+2)*(x+1) ), x), x)
    Out[9]:
           1
    ───────────────
    (2 + x)*(1 + x)


.. index:: calculus

Calculus
========

.. index:: limits

Limits
------

Limits are easy to use in sympy, they follow the syntax limit(function,
variable, point), so to compute the limit of f(x) as x -> 0, you would issue
limit(f, x, 0)::

   >>> from sympy import *
   >>> x=Symbol("x")
   >>> limit(sin(x)/x, x, 0)
   1

you can also calculate the limit at infinity::

   >>> limit(x, x, oo)
   oo

   >>> limit(1/x, x, oo)
   0

   >>> limit(x**x, x, 0)
   1

for some non-trivial examples on limits, you can read the test file
`test_demidovich.py
<http://git.sympy.org/?p=sympy.git;a=blob;f=sympy/series/tests/test_demidovich.py>`_

.. index:: differentiation, diff

Differentiation
---------------

You can differentiate any SymPy expression using ``diff(func, var)``. Examples::

    >>> from sympy import *
    >>> x = Symbol('x')
    >>> diff(sin(x), x)
    cos(x)
    >>> diff(sin(2*x), x)
    2*cos(2*x)

    >>> diff(tan(x), x)
    1 + tan(x)**2

You can check, that it is correct by::

    >>> limit((tan(x+y)-tan(x))/y, y, 0)
    1 + tan(x)**2

Higher derivatives can be calculated using the ``diff(func, var, n)`` method::

    >>> diff(sin(2*x), x, 1)
    2*cos(2*x)

    >>> diff(sin(2*x), x, 2)
    -4*sin(2*x)

    >>> diff(sin(2*x), x, 3)
    -8*cos(2*x)


.. index::
    single: series expansion
    single: expansion; series

Exercises
---------

  1. 
  2.


Series expansion
----------------

Use ``.series(var, point, order)``::

    >>> from sympy import *
    >>> x = Symbol('x')
    >>> cos(x).series(x, 0, 10)
    1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**10)
    >>> (1/cos(x)).series(x, 0, 10)
    1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + 277*x**8/8064 + O(x**10)

Another simple example::

    from sympy import Integral, Symbol, pprint

    x = Symbol("x")
    y = Symbol("y")

    e = 1/(x + y)
    s = e.series(x, 0, 5)

    print(s)
    pprint(s)

That should print the following after the execution::

    1/y + x**2*y**(-3) + x**4*y**(-5) - x*y**(-2) - x**3*y**(-4) + O(x**5)
         2    4         3
    1   x    x    x    x
    ─ + ── + ── - ── - ── + O(x**5)
    y    3    5    2    4
        y    y    y    y

.. index:: integration

Integration
-----------

SymPy has support for indefinite and definite integration of transcendental
elementary and special functions via `integrate()` facility, which uses
powerful extended Risch-Norman algorithm and some heuristics and pattern
matching::

    >>> from sympy import *
    >>> x, y = symbols('xy')

You can integrate elementary functions::

    >>> integrate(6*x**5, x)
    x**6
    >>> integrate(sin(x), x)
    -cos(x)
    >>> integrate(log(x), x)
    -x + x*log(x)
    >>> integrate(2*x + sinh(x), x)
    cosh(x) + x**2

Also special functions are handled easily::

    >>> integrate(exp(-x**2)*erf(x), x)
    pi**(1/2)*erf(x)**2/4

It is possible to compute definite integral::

    >>> integrate(x**3, (x, -1, 1))
    0
    >>> integrate(sin(x), (x, 0, pi/2))
    1
    >>> integrate(cos(x), (x, -pi/2, pi/2))
    2

Also improper integrals are supported as well::

    >>> integrate(exp(-x), (x, 0, oo))
    1
    >>> integrate(log(x), (x, 0, 1))
    -1


.. index:: equations; algebraic, solve

Algebraic equations
-------------------
SymPy is able to solve algebraic equations, in one and several variables.

In ``isympy``::

    In [7]: solve(x**4 - 1, x)
    Out[7]: [ⅈ, 1, -1, -ⅈ]

    In [8]: solve([x + 5*y - 2, -3*x + 6*y - 15], [x, y])
    Out[8]: {y: 1, x: -3}


.. index:: linear algebra

Linear Algebra
==============

.. index:: Matrix

Matrices
--------

Matrices are created as instances from the Matrix class::

    >>> from sympy import Matrix
    >>> Matrix([[1,0], [0,1]])
    [1, 0]
    [0, 1]

unline a numpy array, you can also put Symbols in it::

    >>> x = Symbol('x')
    >>> y = Symbol('y')
    >>> A = Matrix([[1,x], [y,1]])
    >>> A
    [1, x]
    [y, 1]

    >>> A**2
    [1 + x*y,     2*x]
    [    2*y, 1 + x*y]




.. index:: equations; differential, diff, dsolve

Differential Equations
----------------------

SymPy is capable of solving (some) Ordinary Differential
Equations. sympy.ode.dsolve works like this ::

    In [4]: f(x).diff(x, x) + f(x)
    Out[4]:
       2
      d
    ─────(f(x)) + f(x)
    dx dx

    In [5]: dsolve(f(x).diff(x, x) + f(x), f(x))
    Out[5]: C₁*sin(x) + C₂*cos(x)



TODO: more on this, current status of the ODE solver, PDES ??




.. _printing-tutorial:

Printing
========

There are many ways how expressions can be printed.

**Standard**

This is what ``str(expression)`` returns and it looks like this:

    >>> from sympy import Integral
    >>> from sympy.abc import x
    >>> print x**2
    x**2
    >>> print 1/x
    1/x
    >>> print Integral(x**2, x)
    Integral(x**2, x)
    >>>


**Pretty printing**

This is a nice ascii-art printing produced by a ``pprint`` function:

    >>> from sympy import Integral, pprint
    >>> from sympy.abc import x
    >>> pprint(x**2) #doctest: +NORMALIZE_WHITESPACE
     2
    x
    >>> pprint(1/x)
    1
    -
    x
    >>> pprint(Integral(x**2, x))
      /     
     |      
     |  2   
     | x  dx
     |      
    /       


See also the wiki `Pretty Printing
<http://wiki.sympy.org/wiki/Pretty_Printing>`_ for more examples of a nice
unicode printing.

Tip: To make the pretty printing default in the python interpreter, use::

    $ python
    Python 2.5.2 (r252:60911, Jun 25 2008, 17:58:32) 
    [GCC 4.3.1] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from sympy import *
    >>> import sys
    >>> sys.displayhook = pprint
    >>> var("x")
    x
    >>> x**3/3
     3
    x 
    --
    3 
    >>> Integral(x**2, x) #doctest: +NORMALIZE_WHITESPACE
      /     
     |      
     |  2   
     | x  dx
     |      
    /     


**Python printing**

    >>> from sympy.printing.python import python
    >>> from sympy import Integral
    >>> from sympy.abc import x
    >>> print python(x**2)
    x = Symbol('x')
    e = x**2
    >>> print python(1/x)
    x = Symbol('x')
    e = 1/x
    >>> print python(Integral(x**2, x))
    x = Symbol('x')
    e = Integral(x**2, x)


**LaTeX printing**

    >>> from sympy import Integral, latex
    >>> from sympy.abc import x
    >>> latex(x**2)
    x^{2}
    >>> latex(x**2, mode='inline')
    $x^{2}$
    >>> latex(x**2, mode='equation')
    \begin{equation}x^{2}\end{equation}
    >>> latex(x**2, mode='equation*')
    \begin{equation*}x^{2}\end{equation*}
    >>> latex(1/x)
    \frac{1}{x}
    >>> latex(Integral(x**2, x))
    \int x^{2}\,dx
    >>>

**MathML**

::

    >>> from sympy.printing.mathml import mathml
    >>> from sympy import Integral, latex
    >>> from sympy.abc import x
    >>> print mathml(x**2)
    <apply><power/><ci>x</ci><cn>2</cn></apply>
    >>> print mathml(1/x)
    <apply><power/><ci>x</ci><cn>-1</cn></apply>

**Pyglet**

    >>> from sympy import Integral, preview
    >>> from sympy.abc import x
    >>> preview(Integral(x**2, x)) #doctest:+SKIP

And a pyglet window with the LaTeX rendered expression will popup:

.. image:: pics/pngview1.png

Notes
-----

``isympy`` calls ``pprint`` automatically, so that's why you see pretty
printing by default.

Note that there is also a printing module available, ``sympy.printing``.  Other
printing methods available trough this module are:
