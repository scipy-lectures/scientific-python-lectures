First steps
-------------


Start the **Ipython** shell (an enhanced interactive Python shell):

* by typing "ipython" from a Linux/Mac terminal, or from the Windows cmd shell,
* **or** by starting the program from a menu, e.g. in the `Python(x,y)`_ or
  `EPD`_ menu if you have installed one of these scientific-Python suites.

.. _`Python(x,y)`: http://www.pythonxy.com/
.. _`EPD`: http://www.enthought.com/products/epd.php

If you don't have Ipython installed on your computer, other Python shells
are available, such as the plain Python shell started by typing "python"
in a terminal, or the Idle interpreter. However, we advise to use the
Ipython shell because of its enhanced features, especially for
interactive scientific computing.

Once you have started the interpreter, type ::

    >>> print "Hello, world!"
    Hello, world!

The message "Hello, world!" is then displayed. You just executed your
first Python instruction, congratulations!

To get yourself started, type the following stack of instructions ::

    >>> a = 3
    >>> b = 2*a
    >>> type(b)
    <type 'int'>
    >>> print b
    6
    >>> a*b 
    18
    >>> b = 'hello' 
    >>> type(b)
    <type 'str'>
    >>> b + b
    'hellohello'
    >>> 2*b
    'hellohello'

Two variables ``a`` and ``b`` have been defined above. Note that one does
not declare the type of an variable before assigning its value. In C,
conversely, one should write:

.. sourcecode:: c

    int a = 3;

In addition, the type of a variable may change, in the sense that at
one point in time it can be equal to a value of a certain type, and a
second point in time, it can be equal to a value of a different
type. `b` was first equal to an integer, but it became equal to a
string when it was assigned the value `'hello'`. Operations on
integers (``b=2*a``) are coded natively in Python, and so are some
operations on strings such as additions and multiplications, which
amount respectively to concatenation and repetition.
