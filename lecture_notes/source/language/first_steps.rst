First steps
-------------


Start the **Ipython** shell (an enhanced interactive Python shell):

* by typing "Ipython" from a Linux/Mac terminal, or from the Windows cmd shell,
* **or** by starting the program from a menu, e.g. in the Python(x,y) or
  EPD menu if you have installed one these scientific-Python suites.

.. :ref:`pythonxy`

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

Two objects ``a`` and ``b`` have been defined above. Note that one does
not declare the type of an object before assigning its value. In C,
conversely, one should write:

.. sourcecode:: c

    int a;
    a = 3;

In addition, the type of an object may change. `b` was first an integer,
but it became a string when it was assigned the value `hello`. Operations
on integers (``b=2*a``) are coded natively in the Python standard
library, and so are some operations on strings such as additions and
multiplications, which amount respectively to concatenation and
repetition. 




