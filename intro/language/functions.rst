Defining functions
=====================

Function definition
-------------------

.. sourcecode:: pycon

    >>> def test():
    ...     print('in test function')

    >>> test()
    in test function

.. Warning::

    Function blocks must be indented as other control-flow blocks.

Return statement
----------------

Functions can *optionally* return values.

.. sourcecode:: pycon

    >>> def disk_area(radius):
    ...     return 3.14 * radius * radius

    >>> disk_area(1.5)
    7.0649999999999995

.. Note:: By default, functions return ``None``.

.. Note:: Note the syntax to define a function:

    * the ``def`` keyword;

    * is followed by the function's **name**, then

    * the arguments of the function are given between parentheses followed
      by a colon.

    * the function body;

    * and ``return object`` for optionally returning values.


Parameters
----------

Mandatory parameters (positional arguments)

.. sourcecode:: pycon

    >>> def double_it(x):
    ...     return x * 2

    >>> double_it(3)
    6

    >>> double_it()
    ---------------------------------------------------------------------------
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: double_it() takes exactly 1 argument (0 given)

Optional parameters (keyword or named arguments)

.. sourcecode:: pycon

    >>> def double_it(x=2):
    ...     return x * 2

    >>> double_it()
    4

    >>> double_it(3)
    6

Keyword arguments allow you to specify *default values*.

.. warning::

   Default values are evaluated when the function is defined, not when
   it is called. This can be problematic when using mutable types (e.g.
   dictionary or list) and modifying them in the function body, since the
   modifications will be persistent across invocations of the function.

   Using an immutable type in a keyword argument:

   .. sourcecode:: pycon

       >>> bigx = 10

       >>> def double_it(x=bigx):
       ...     return x * 2

       >>> bigx = 1e9  # Now really big

       >>> double_it()
       20

   Using an mutable type in a keyword argument (and modifying it inside the
   function body):

   .. sourcecode:: pycon

       >>> def add_to_dict(args={'a': 1, 'b': 2}):
       ...     for i in args.keys():
       ...         args[i] += 1
       ...     print args

       >>> add_to_dict    # doctest: +ELLIPSIS
       <function add_to_dict at 0x...>

       >>> add_to_dict()
       {'a': 2, 'b': 3}

       >>> add_to_dict()
       {'a': 3, 'b': 4}

       >>> add_to_dict()
       {'a': 4, 'b': 5}

.. tip::

  More involved example implementing python's slicing:

  .. sourcecode:: pycon

    >>> def slicer(seq, start=None, stop=None, step=None):
    ...     """Implement basic python slicing."""
    ...     return seq[start:stop:step]

    >>> rhyme = 'one fish, two fish, red fish, blue fish'.split()

    >>> rhyme
    ['one', 'fish,', 'two', 'fish,', 'red', 'fish,', 'blue', 'fish']

    >>> slicer(rhyme)
    ['one', 'fish,', 'two', 'fish,', 'red', 'fish,', 'blue', 'fish']

    >>> slicer(rhyme, step=2)
    ['one', 'two', 'red', 'blue']

    >>> slicer(rhyme, 1, step=2)
    ['fish,', 'fish,', 'fish,', 'fish']

    >>> slicer(rhyme, start=1, stop=4, step=2)
    ['fish,', 'fish,']

  The order of the keyword arguments does not matter:

  .. sourcecode:: pycon

    >>> slicer(rhyme, step=2, start=1, stop=4)
    ['fish,', 'fish,']

  but it is good practice to use the same ordering as the function's
  definition.

*Keyword arguments* are a very convenient feature for defining functions
with a variable number of arguments, especially when default values are
to be used in most calls to the function.

Passing by value
----------------

.. tip::

    Can you modify the value of a variable inside a function? Most languages
    (C, Java, ...) distinguish "passing by value" and "passing by reference".
    In Python, such a distinction is somewhat artificial, and it is a bit
    subtle whether your variables are going to be modified or not.
    Fortunately, there exist clear rules.

    Parameters to functions are references to objects, which are passed by
    value. When you pass a variable to a function, python passes the
    reference to the object to which the variable refers (the **value**).
    Not the variable itself.

If the **value** passed in a function is immutable, the function does not
modify the caller's variable.  If the **value** is mutable, the function
may modify the caller's variable in-place::

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



Functions have a local variable table called a *local namespace*.

The variable ``x`` only exists within the function ``try_to_modify``.


Global variables
----------------

Variables declared outside the function can be referenced within the
function:

.. sourcecode:: pycon

    >>> x = 5

    >>> def addx(y):
    ...     return x + y

    >>> addx(10)
    15

But these "global" variables cannot be modified within the function,
unless declared **global** in the function.

This doesn't work:

.. sourcecode:: pycon

    >>> def setx(y):
    ...     x = y
    ...     print('x is %d' % x)

    >>> setx(10)
    x is 10

    >>> x
    5

This works:

.. sourcecode:: pycon

    >>> def setx(y):
    ...     global x
    ...     x = y
    ...     print('x is %d' % x)

    >>> setx(10)
    x is 10

    >>> x
    10


Variable number of parameters
-----------------------------
Special forms of parameters:
  * ``*args``: any number of positional arguments packed into a tuple
  * ``**kwargs``: any number of keyword arguments packed into a dictionary

.. sourcecode:: pycon

    >>> def variable_args(*args, **kwargs):
    ...     print 'args is', args
    ...     print 'kwargs is', kwargs

    >>> variable_args('one', 'two', x=1, y=2, z=3)
    args is ('one', 'two')
    kwargs is {'y': 2, 'x': 1, 'z': 3}


Docstrings
----------

Documentation about what the function does and its parameters.  General
convention:

.. sourcecode:: pycon

    In [67]: def funcname(params):
       ....:     """Concise one-line sentence describing the function.
       ....:
       ....:     Extended summary which can contain multiple paragraphs.
       ....:     """
       ....:     # function body
       ....:     pass
       ....:

    In [68]: funcname?
    Type:           function
    Base Class:     type 'function'>
    String Form:    <function funcname at 0xeaa0f0>
    Namespace:      Interactive
    File:           <ipython console>
    Definition:     funcname(params)
    Docstring:
        Concise one-line sentence describing the function.

        Extended summary which can contain multiple paragraphs.

.. Note:: **Docstring guidelines**


    For the sake of standardization, the `Docstring
    Conventions <http://www.python.org/dev/peps/pep-0257>`_ webpage
    documents the semantics and conventions associated with Python
    docstrings.

    Also, the Numpy and Scipy modules have defined a precise standard
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

.. sourcecode:: pycon

    >>> va = variable_args

    >>> va('three', x=1, y=2)
    args is ('three',)
    kwargs is {'y': 2, 'x': 1}


Methods
-------

Methods are functions attached to objects.  You've seen these in our
examples on *lists*, *dictionaries*, *strings*, etc...


Exercises
---------

.. topic:: Exercise: Fibonacci sequence
    :class: green

    Write a function that displays the ``n`` first terms of the Fibonacci
    sequence, defined by:

    * ``u_0 = 1; u_1 = 1``
    * ``u_(n+2) = u_(n+1) + u_n``

.. :ref:`fibonacci`

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
