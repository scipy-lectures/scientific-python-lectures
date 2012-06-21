Defining functions
=====================

Function definition
-------------------

.. sourcecode:: ipython

    In [56]: def test():
       ....:     print('in test function')
       ....:     
       ....:     

    In [57]: test()
    in test function

.. Warning:: 

    Function blocks must be indented as other control-flow blocks.

Return statement
----------------

Functions can *optionally* return values.

.. sourcecode:: ipython

    In [6]: def disk_area(radius):
       ...:     return 3.14 * radius * radius
       ...: 

    In [8]: disk_area(1.5)
    Out[8]: 7.0649999999999995

.. Note:: By default, functions return ``None``.

.. Note:: Note the syntax to define a function:

    * the ``def`` keyword;

    * is followed by the function's **name**, then

    * the arguments of the function are given between brackets followed
      by a colon.

    * the function body ;

    * and ``return object`` for optionally returning values.


Parameters
----------

Mandatory parameters (positional arguments)

.. sourcecode:: ipython

    In [81]: def double_it(x):
       ....:     return x * 2
       ....: 

    In [82]: double_it(3)
    Out[82]: 6

    In [83]: double_it()
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/<ipython console> in <module>()

    TypeError: double_it() takes exactly 1 argument (0 given)

Optional parameters (keyword or named arguments)

.. sourcecode:: ipython

    In [84]: def double_it(x=2):
       ....:     return x * 2
       ....: 

    In [85]: double_it()
    Out[85]: 4

    In [86]: double_it(3)
    Out[86]: 6

Keyword arguments allow you to specify *default values*.

.. warning:: 

   Default values are evaluated when the function is defined, not when
   it is called.

.. sourcecode:: ipython

    In [124]: bigx = 10

    In [125]: def double_it(x=bigx):
       .....:     return x * 2
       .....: 

    In [126]: bigx = 1e9  # Now really big

    In [128]: double_it()
    Out[128]: 20

More involved example implementing python's slicing:

.. sourcecode:: ipython

    In [98]: def slicer(seq, start=None, stop=None, step=None):
       ....:     """Implement basic python slicing."""
       ....:     return seq[start:stop:step]
       ....: 

    In [101]: rhyme = 'one fish, two fish, red fish, blue fish'.split()

    In [102]: rhyme
    Out[102]: ['one', 'fish,', 'two', 'fish,', 'red', 'fish,', 'blue', 'fish']

    In [103]: slicer(rhyme)
    Out[103]: ['one', 'fish,', 'two', 'fish,', 'red', 'fish,', 'blue', 'fish']

    In [104]: slicer(rhyme, step=2)
    Out[104]: ['one', 'two', 'red', 'blue']

    In [105]: slicer(rhyme, 1, step=2)
    Out[105]: ['fish,', 'fish,', 'fish,', 'fish']

    In [106]: slicer(rhyme, start=1, stop=4, step=2)
    Out[106]: ['fish,', 'fish,']

The order of the keyword arguments does not matter:

.. sourcecode:: ipython

    In [107]: slicer(rhyme, step=2, start=1, stop=4)
    Out[107]: ['fish,', 'fish,']

but it is good practice to use the same ordering as the function's
definition.

*Keyword arguments* are a very convenient feature for defining functions
with a variable number of arguments, especially when default values are
to be used in most calls to the function.

Passed by value
---------------

Can you modify the value of a variable inside a function? Most languages
(C, Java, ...) distinguish "passing by value" and "passing by reference".
In Python, such a distinction is somewhat artificial, and it is a bit
subtle whether your variables are going to be modified or not.
Fortunately, there exist clear rules.

Parameters to functions are references to objects, which are passed by
value. When you pass a variable to a function, python passes the
reference to the object to which the variable refers (the **value**).
Not the variable itself.

If the **value** is immutable, the function does not modify the caller's
variable.  If the **value** is mutable, the function may modify the
caller's variable in-place::

    >>> def try_to_modify(x, y, z):
    ...     x = 23
    ...     y.append(42)
    ...     z = [99] # new reference
    ...     print(x)
    ...     print(y)
    ...     print(z)
    ...     
    >>> a = 77    # immutable variable
    >>> b = [99]  # mutable variable
    >>> c = [28]
    >>> try_to_modify(a, b, c)
    23
    [99, 42]
    [99]
    >>> print(a)
    77
    >>> print(b)
    [99, 42]
    >>> print(c)
    [28]



Functions have a local variable table. Called a *local namespace*.

The variable ``x`` only exists within the function *foo*.


Global variables
----------------

Variables declared outside the function can be referenced within the
function:

.. sourcecode:: ipython

    In [114]: x = 5

    In [115]: def addx(y):
       .....:     return x + y
       .....: 

    In [116]: addx(10)
    Out[116]: 15

But these "global" variables cannot be modified within the function,
unless declared **global** in the function.

This doesn't work:

.. sourcecode:: ipython

    In [117]: def setx(y):
       .....:     x = y
       .....:     print('x is %d' % x)
       .....:     
       .....:     

    In [118]: setx(10)
    x is 10

    In [120]: x
    Out[120]: 5

This works:

.. sourcecode:: ipython

    In [121]: def setx(y):
       .....:     global x
       .....:     x = y
       .....:     print('x is %d' % x)
       .....:     
       .....:     

    In [122]: setx(10)
    x is 10

    In [123]: x
    Out[123]: 10


Variable number of parameters
-----------------------------
Special forms of parameters:
  * \*args: any number of positional arguments packed into a tuple
  * \**kwargs: any number of keyword arguments packed into a dictionary

.. sourcecode:: ipython

    In [35]: def variable_args(*args, **kwargs):
       ....:     print 'args is', args
       ....:     print 'kwargs is', kwargs
       ....: 

    In [36]: variable_args('one', 'two', x=1, y=2, z=3)
    args is ('one', 'two')
    kwargs is {'y': 2, 'x': 1, 'z': 3}


Docstrings
----------

Documentation about what the function does and it's parameters.  General
convention:

.. sourcecode:: ipython

    In [67]: def funcname(params):
       ....:     """Concise one-line sentence describing the function.
       ....: 
       ....:     Extended summary which can contain multiple paragraphs.
       ....:     """
       ....:     # function body
       ....:     pass
       ....: 

    In [68]: funcname?
    Type:		function
    Base Class:	<type 'function'>
    String Form:	<function funcname at 0xeaa0f0>
    Namespace:	Interactive
    File:		/Users/cburns/src/scipy2009/.../<ipython console>
    Definition:	funcname(params)
    Docstring:
        Concise one-line sentence describing the function.

        Extended summary which can contain multiple paragraphs.

.. Note:: **Docstring guidelines**


    For the sake of standardization, the `Docstring
    Conventions <http://www.python.org/dev/peps/pep-0257>`_ webpage
    documents the semantics and conventions associated with Python
    docstrings.

    Also, the Numpy and Scipy modules have defined a precised standard
    for documenting scientific functions, that you may want to follow for
    your own functions, with a ``Parameters`` section, an ``Examples``
    section, etc. See
    http://projects.scipy.org/numpy/wiki/CodingStyleGuidelines#docstring-standard 
    and http://projects.scipy.org/numpy/browser/trunk/doc/example.py#L37

Functions are objects
---------------------
Functions are first-class objects, which means they can be:
  * assigned to a variable
  * an item in a list (or any collection)
  * passed as an argument to another function.

.. sourcecode:: ipython

    In [38]: va = variable_args

    In [39]: va('three', x=1, y=2)
    args is ('three',)
    kwargs is {'y': 2, 'x': 1}


Methods
-------

Methods are functions attached to objects.  You've seen these in our
examples on **lists**, **dictionaries**, **strings**, etc...


Exercises
---------

.. topic:: Exercise: Quicksort
    :class: green

    Implement the quicksort algorithm, as defined by wikipedia::

	function quicksort(array)
	    var list less, greater
	    if length(array) < 2  
		return array  
	    select and remove a pivot value pivot from array
	    for each x in array
		if x < pivot + 1 then append x to less
		else append x to greater
	    return concatenate(quicksort(less), pivot, quicksort(greater))

.. :ref:`quick_sort`

.. topic:: Exercise: Fibonacci sequence
    :class: green

    Write a function that displays the ``n`` first terms of the Fibonacci
    sequence, defined by:

    * ``u_0 = 1; u_1 = 1``
    * ``u_(n+2) = u_(n+1) + u_n``

.. :ref:`fibonacci`
