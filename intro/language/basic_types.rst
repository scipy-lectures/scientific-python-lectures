Basic types
============

Numerical types
----------------

:Integer:

    >>> 1 + 1
    2
    >>> a = 4

:Floats:

    >>> c = 2.1

:Complex:

    >>> a = 1.5 + 0.5j
    >>> a.real
    1.5
    >>> a.imag
    0.5

:Booleans:

    >>> 3 > 4
    False
    >>> test = (3 > 4)
    >>> test
    False
    >>> type(test)
    <type 'bool'>



A Python shell can therefore replace your pocket calculator, with the
basic arithmetic operations ``+``, ``-``, ``*``, ``/``, ``%`` (modulo)
natively implemented::

    >>> 7 * 3.
    21.0
    >>> 2**10
    1024
    >>> 8 % 3
    2

.. warning:: Integer division
    ::

	>>> 3 / 2
	1

    **Trick**: use floats:: 

	>>> 3 / 2.
	1.5

	>>> a = 3
	>>> b = 2
	>>> a / b
	1
	>>> a / float(b)
	1.5


* Scalar types: int, float, complex, bool::

    >>> type(1)
    <type 'int'>
    >>> type(1.)
    <type 'float'>
    >>> type(1. + 0j )
    <type 'complex'>

    >>> a = 3
    >>> type(a)
    <type 'int'>



* Type conversion::

    >>> float(1)
    1.0

Containers
------------

Python provides many efficient types of containers, in which collections of
objects can be stored.

Lists
~~~~~


A list is an ordered collection of objects, that may have different
types. For example ::

    >>> l = [1, 2, 3, 4, 5]
    >>> type(l)
    <type 'list'>

* Indexing: accessing individual objects contained in the list::

    >>> l[2]
    3

  Counting from the end with negative indices::

    >>> l[-1]
    5
    >>> l[-2]
    4

.. warning::

    **Indexing starts at 0** (as in C), not at 1 (as in Fortran or Matlab)!

* Slicing: obtaining sublists of regularly-spaced elements

::

    >>> l
    [1, 2, 3, 4, 5]
    >>> l[2:4]
    [3, 4]

.. Warning::

    Note that ``l[start:stop]`` contains the elements with indices ``i``
    such as  ``start<= i < stop`` (``i`` ranging from ``start`` to
    ``stop-1``). Therefore, ``l[start:stop]`` has ``(stop-start)`` elements.

**Slicing syntax**: `l[start:stop:stride]`

All slicing parameters are optional::

    >>> l[3:]
    [4, 5]
    >>> l[:3]
    [1, 2, 3]
    >>> l[::2]
    [1, 3, 5]

Lists are *mutable* objects and can be modified::

    >>> l[0] = 28
    >>> l
    [28, 2, 3, 4, 5]
    >>> l[2:4] = [3, 8] 
    >>> l
    [28, 2, 3, 8, 5]

.. Note::

    The elements of a list may have different types::

	>>> l = [3, 2, 'hello']
	>>> l
	[3, 2, 'hello']
	>>> l[1], l[2]
	(2, 'hello')

    For collections of numerical data that all have the same type, it
    is often **more efficient** to use the ``array`` type provided by
    the ``numpy`` module. A NumPy array is a chunk of memory
    containing fixed-sized items.  With NumPy arrays, operations on
    elements can be faster because elements are regularly spaced in
    memory and more operations are perfomed through specialized C
    functions instead of Python loops.


Python offers a large panel of functions to modify lists,
or query them. Here are a few examples; for more details, see
http://docs.python.org/tutorial/datastructures.html#more-on-lists

Add and remove elements::

    >>> l = [1, 2, 3, 4, 5]
    >>> l.append(6)
    >>> l
    [1, 2, 3, 4, 5, 6]
    >>> l.pop()
    6
    >>> l
    [1, 2, 3, 4, 5]
    >>> l.extend([6, 7]) # extend l, in-place
    >>> l
    [1, 2, 3, 4, 5, 6, 7]
    >>> l = l[:-2]
    >>> l
    [1, 2, 3, 4, 5]


Reverse `l`::

    >>> r = l[::-1]
    >>> r
    [5, 4, 3, 2, 1]

Concatenate and repeat lists:: 

    >>> r + l
    [5, 4, 3, 2, 1, 1, 2, 3, 4, 5]
    >>> 2 * r
    [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]

Sort r (in-place)::

    >>> r.sort()
    >>> r
    [1, 2, 3, 4, 5]


.. Note:: **Methods and Object-Oriented Programming**

    The notation ``r.method()`` (``r.sort(), r.append(3), l.pop()``) is our
    first example of object-oriented programming (OOP). Being a ``list``, the
    object `r` owns the *method* `function` that is called using the notation
    **.**. No further knowledge of OOP than understanding the notation **.** is
    necessary for going through this tutorial.  


.. note:: **Discovering methods:**

    In IPython: tab-completion (press tab)

    .. sourcecode:: ipython

        In [28]: r.
        r.__add__           r.__iadd__          r.__setattr__
        r.__class__         r.__imul__          r.__setitem__
        r.__contains__      r.__init__          r.__setslice__
        r.__delattr__       r.__iter__          r.__sizeof__
        r.__delitem__       r.__le__            r.__str__
        r.__delslice__      r.__len__           r.__subclasshook__
        r.__doc__           r.__lt__            r.append
        r.__eq__            r.__mul__           r.count
        r.__format__        r.__ne__            r.extend
        r.__ge__            r.__new__           r.index
        r.__getattribute__  r.__reduce__        r.insert
        r.__getitem__       r.__reduce_ex__     r.pop
        r.__getslice__      r.__repr__          r.remove
        r.__gt__            r.__reversed__      r.reverse
        r.__hash__          r.__rmul__          r.sort




Strings
~~~~~~~ 

Different string syntaxes (simple, double or triple quotes)::

    s = 'Hello, how are you?'
    s = "Hi, what's up"
    s = '''Hello,                 # tripling the quotes allows the
           how are you'''         # the string to span more than one line
    s = """Hi,
	   what's up?"""

.. sourcecode:: ipython

    In [1]: 'Hi, what's up?'
    ------------------------------------------------------------
       File "<ipython console>", line 1
	 'Hi, what's up?'
               ^
    SyntaxError: invalid syntax


The newline character is ``\n``, and the tab character is
``\t``.

Strings are collections like lists. Hence they can be indexed and sliced,
using the same syntax and rules.

Indexing::

    >>> a = "hello"
    >>> a[0]
    'h'
    >>> a[1]
    'e'
    >>> a[-1]
    'o'


(Remember that negative indices correspond to counting from the right
end.)

Slicing::


    >>> a = "hello, world!"
    >>> a[3:6] # 3rd to 6th (excluded) elements: elements 3, 4, 5
    'lo,'
    >>> a[2:10:2] # Syntax: a[start:stop:step]
    'lo o'
    >>> a[::3] # every three characters, from beginning to end 
    'hl r!'

Accents and special characters can also be handled in Unicode strings (see
http://docs.python.org/tutorial/introduction.html#unicode-strings).


A string is an **immutable object** and it is not possible to modify its
contents. One may however create new strings from the original one.

.. sourcecode:: ipython

    In [53]: a = "hello, world!"
    In [54]: a[2] = 'z'
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call
    last)

    /home/gouillar/travail/sgr/2009/talks/dakar_python/cours/gael/essai/source/<ipython
    console> in <module>()

    TypeError: 'str' object does not support item assignment
    In [55]: a.replace('l', 'z', 1)
    Out[55]: 'hezlo, world!'
    In [56]: a.replace('l', 'z')
    Out[56]: 'hezzo, worzd!'

Strings have many useful methods, such as ``a.replace`` as seen above.
Remember the ``a.`` object-oriented notation and use tab completion or
``help(str)`` to search for new methods.

.. Note:: 

    Python offers advanced possibilities for manipulating strings,
    looking for patterns or formatting. Due to lack of time this topic is
    not addressed here, but the interested reader is referred to
    http://docs.python.org/library/stdtypes.html#string-methods and
    http://docs.python.org/library/string.html#new-string-formatting

* String substitution::

    >>> 'An integer: %i; a float: %f; another string: %s' % (1, 0.1, 'string')
    'An integer: 1; a float: 0.100000; another string: string'

    >>> i = 102
    >>> filename = 'processing_of_dataset_%03d.txt'%i
    >>> filename
    'processing_of_dataset_102.txt'


Dictionaries
~~~~~~~~~~~~~

A dictionary is basically an efficient table that **maps keys to
values**. It is an **unordered** container::


    >>> tel = {'emmanuelle': 5752, 'sebastian': 5578}
    >>> tel['francis'] = 5915 
    >>> tel
    {'sebastian': 5578, 'francis': 5915, 'emmanuelle': 5752}
    >>> tel['sebastian']
    5578
    >>> tel.keys()
    ['sebastian', 'francis', 'emmanuelle']
    >>> tel.values()
    [5578, 5915, 5752]
    >>> 'francis' in tel
    True

It can be used to conveniently store and retrieve values
associated with a name (a string for a date, a name, etc.). See
http://docs.python.org/tutorial/datastructures.html#dictionaries
for more information.

A dictionary can have keys (resp. values) with different types::

    >>> d = {'a':1, 'b':2, 3:'hello'}
    >>> d
    {'a': 1, 3: 'hello', 'b': 2}

More container types
~~~~~~~~~~~~~~~~~~~~

* **Tuples**

Tuples are basically immutable lists. The elements of a tuple are written
between parentheses, or just separated by commas::


    >>> t = 12345, 54321, 'hello!'
    >>> t[0]
    12345
    >>> t
    (12345, 54321, 'hello!')
    >>> u = (0, 2)

* **Sets:** unordered, unique items::

    >>> s = set(('a', 'b', 'c', 'a'))
    >>> s
    set(['a', 'c', 'b'])
    >>> s.difference(('a', 'b'))
    set(['c'])

.. topic:: A bag of Ipython tricks

    * Several Linux shell commands work in Ipython, such as ``ls``,
    * ``pwd``,
      ``cd``, etc.

    * To get help about objects, functions, etc., type ``help object``.
      Just type help() to get started.

    * Use **tab-completion** as much as possible: while typing the
      beginning of an object's name (variable, function, module), press 
      the **Tab** key and Ipython will complete the expression to match 
      available names. If many names are possible, a list of names is 
      displayed.

    * **History**: press the `up` (resp. `down`) arrow to go through all
      previous (resp. next) instructions starting with the expression on
      the left of the cursor (put the cursor at the beginning of the line
      to go through all previous commands) 

    * You may log your session by using the Ipython "magic command"
      %logstart. Your instructions will be saved in a file, that you can
      execute as a script in a different session.


.. sourcecode:: ipython

    In [1]: %logstart commands.log
    Activating auto-logging. Current session state plus future input
    saved.
    Filename       : commands.log
    Mode           : backup
    Output logging : False
    Raw input log  : False
    Timestamping   : False
    State          : active

