Control Flow
============

Controls the order in which the code is executed.

if/elif/else
------------

.. sourcecode:: ipython
  
    In [1]: if 2**2 == 4:
       ...:     print('Obvious!')
       ...: 
    Obvious!


**Blocks are delimited by indentation**

Type the following lines in your Python interpreter, and be careful to
**respect the indentation depth**. The Ipython shell automatically
increases the indentation depth after a column ``:`` sign; to
decrease the indentation depth, go four spaces to the left with the
Backspace key. Press the Enter key twice to leave the logical block.

.. sourcecode:: ipython

    In [2]: a = 10
    
    In [3]: if a == 1:
       ...:     print(1)
       ...: elif a == 2:
       ...:     print(2)
       ...: else:
       ...:     print('A lot')
       ...: 
    A lot

Indentation is compulsory in scripts as well. As an exercise, re-type the
previous lines with the same indentation in a script ``condition.py``, and
execute the script with ``run condition.py`` in Ipython.

for/range
----------

Iterating with an index:

.. sourcecode:: ipython

    In [4]: for i in range(4):
       ...:     print(i)
       ...: 
    0
    1
    2
    3

But most often, it is more readable to iterate over values:

.. sourcecode:: ipython

    In [5]: for word in ('cool', 'powerful', 'readable'):
       ...:     print('Python is %s' % word)
       ...: 
    Python is cool
    Python is powerful
    Python is readable


while/break/continue
---------------------

Typical C-style while loop (Mandelbrot problem):

.. sourcecode:: ipython

    In [6]: z = 1 + 1j

    In [7]: while abs(z) < 100:
       ...:     z = z**2 + 1
       ...:     

    In [8]: z
    Out[8]: (-134+352j)

**More advanced features**

``break`` out of enclosing for/while loop:

.. sourcecode:: ipython

    In [9]: z = 1 + 1j

    In [10]: while abs(z) < 100:
       ....:     if z.imag == 0:
       ....:         break
       ....:     z = z**2 + 1
       ....:     
       ....:     


``continue`` the next iteration of a loop.::

    >>> a = [1, 0, 2, 4]
    >>> for element in a:
    ...     if element == 0:
    ...         continue
    ...     print 1. / element
    ...     
    1.0
    0.5
    0.25



Conditional Expressions
-----------------------

* `if object`

  Evaluates to True:
    * any non-zero value
    * any sequence with a length > 0

  Evaluates to False:
    * any zero value
    * any empty sequence

* `a == b`

  Tests equality, with logics:

  .. sourcecode:: ipython

    In [19]: 1 == 1.
    Out[19]: True

* `a is b`

  Tests identity: both objects are the same

  .. sourcecode:: ipython

    In [20]: 1 is 1.
    Out[20]: False

    In [21]: a = 1

    In [22]: b = 1

    In [23]: a is b
    Out[23]: True

* `a in b`

  For any collection `b`: `b` contains `a` ::

    >>> b = [1, 2, 3]
    >>> 2 in b
    True
    >>> 5 in b
    False


  If `b` is a dictionary, this tests that `a` is a key of `b`.


Advanced iteration
-------------------------

Iterate over any *sequence*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* You can iterate over any sequence (string, list, dictionary, file, ...)

  .. sourcecode:: ipython

    In [11]: vowels = 'aeiouy'

    In [12]: for i in 'powerful':
       ....:     if i in vowels:
       ....:         print(i),
       ....:         
       ....:         
    o e u

::

    >>> message = "Hello how are you?"
    >>> message.split() # returns a list
    ['Hello', 'how', 'are', 'you?']
    >>> for word in message.split():
    ...     print word
    ...     
    Hello
    how
    are
    you?

Few languages (in particular, languages for scienfic computing) allow to
loop over anything but integers/indices. With Python it is possible to
loop exactly over the objects of interest without bothering with indices
you often don't care about.
 

.. warning:: Not safe to modify the sequence you are iterating over.

Keeping track of enumeration number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common task is to iterate over a sequence while keeping track of the
item number.

* Could use while loop with a counter as above. Or a for loop:

  .. sourcecode:: ipython

    In [13]: for i in range(0, len(words)):
       ....:     print(i, words[i])
       ....:     
       ....:     
    0 cool
    1 powerful
    2 readable

* But Python provides **enumerate** for this::

    >>> words = ('cool', 'powerful', 'readable')
    >>> for index, item in enumerate(words):
    ...     print index, item
    ...     
    0 cool
    1 powerful
    2 readable



Looping over a dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use **iteritems**:

.. sourcecode:: ipython

    In [15]: d = {'a': 1, 'b':1.2, 'c':1j}

    In [15]: for key, val in d.iteritems():
       ....:     print('Key: %s has value: %s' % (key, val))
       ....:     
       ....:     
    Key: a has value: 1
    Key: c has value: 1j
    Key: b has value: 1.2

List Comprehensions
-------------------

.. sourcecode:: ipython

	In [16]: [i**2 for i in range(4)]
	Out[16]: [0, 1, 4, 9]



.. topic:: Exercise

    Compute the decimals of Pi using the Wallis formula:

    .. image:: pi_formula.png
	:align: center

:ref:`pi_wallis`


.. Note:: **Good practices**

    * **Indentation: no choice!**

    Indenting is compulsory in Python. Every commands block following a
    colon bears an additional indentation level with respect to the
    previous line with a colon. One must therefore indent after 
    ``def f():`` or ``while:``. At the end of such logical blocks, one
    decreases the indentation depth (and re-increases it if a new block is
    entered, etc.)

    Strict respect of indentation is the price to pay for getting rid of
    ``{`` or ``;`` characters that delineate logical blocks in other
    languages. Improper indentation leads to errors such as

    .. sourcecode:: ipython

        ------------------------------------------------------------
        IndentationError: unexpected indent (test.py, line 2)

    All this indentation business can be a bit confusing in the
    beginning. However, with the clear indentation, and in the absence of
    extra characters, the resulting code is very nice to read compared to
    other languages.


    * **Indentation depth**: 

    Inside your text editor, you may choose to
    indent with any positive number of spaces (1, 2, 3, 4, ...). However,
    it is considered good practice to **indent with 4 spaces**. You may
    configure your editor to map the ``Tab`` key to a 4-space
    indentation. In Python(x,y), the editor ``Scite`` is already
    configured this way.	
 
    * **Style guidelines**

    **Long lines**: you should not write very long lines that span over more
    than (e.g.) 80 characters. Long lines can be broken with the ``\``
    character ::
   
        >>> long_line = "Here is a very very long line \
        ... that we break in two parts."

    **Spaces**

    Write well-spaced code: put whitespaces after commas, around arithmetic
    operators, etc.:: 

	>>> a = 1 # yes
	>>> a=1 # too cramped

    A certain number of rules
    for writing "beautiful" code (and more importantly using the same
    conventions as anybody else!) are given in the `Style Guide for Python
    Code <http://www.python.org/dev/peps/pep-0008>`_.

    * Use **meaningful** object **names**

    Self-explaining names improve greatly the readibility of a code. 
 
