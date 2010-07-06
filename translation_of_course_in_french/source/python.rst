A (very short) introduction to Python
=====================================


.. topic:: Python for scientific computing

    We introduce here the Python language. Only the bare minimum
    necessary for getting started with Numpy and Scipy is addressed here.
    To learn more about the language, consider going through the
    excellent tutorial http://docs.python.org/tutorial. Dedicated books
    are also available, such as http://diveintopython.org/.


What is Python?
---------------

.. image:: python-logo.png
   :align: right

Python is a **programming language**, as are C, Fortran, BASIC, PHP,
etc. Some specific features of Python are as follows:

* an *interpreted* (as opposed to *compiled*) language. Contrary to e.g.
  C or Fortran, one does not compile Python code before executing it. In
  addition, Python can be used **interactively**: many Python
  interpreters are available, from which commands and scripts can be
  executed.

* a free software released under an **open-source** license: Python can
  be used and distributed free of charge, even for building commercial
  software.

* **multi-platform**: Python is available for all major operating
  systems, Windows, Linux/Unix, MacOS X, most likely your mobile phone OS,
  etc.

* a very readable language with clear non-verbose syntax

* a language for which a large variety of high-quality packages are
  available for various applications, from web frameworks to scientific
  computing.

* a language very easy to interface with other languages, in particular C
  and C++.

* Some other features of the language are illustrated just below. For
  example, Python is an object-oriented language, with dynamic typing
  (an object's type can change during the course of a program).


See http://www.python.org/about/ for more information about
distinguishing features of Python. 

First steps
-----------

Start the **Ipython** shell (an enhanced interactive Python shell):

* by typing "Ipython" from a Linux/Mac terminal, or from the Windows cmd shell,
* **or** by starting the program from a menu, e.g. in the Python(x,y) or
  EPD menu if you have installed one these scientific-Python suites.

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

.. topic:: A bag of Ipython tricks

    * Several Linux shell commands work in Ipython, such as ``ls``, ``pwd``,
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

    In [1]: %logstart commandes.log
    Activating auto-logging. Current session state plus future input saved.
    Filename       : commandes.log
    Mode           : backup
    Output logging : False
    Raw input log  : False
    Timestamping   : False
    State          : active
 

Different objects
-------------------

**Numerical types**

We have created above integer variables (``int``). There exist also
floats ::

    >>> c = 2.1

and booleans::

    >>> c > a
    False
    >>> test = (c > a)
    >>> test
    False
    >>> type(test)
    <type 'bool'>

Complex numbers are a native type in Python ::

    >>> a=1.5+0.5j
    >>> a.real
    1.5
    >>> a.imag
    0.5

A Python shell can therefore replace your pocket calculator, with the
basic arithmetic operations ``+``, ``-``, ``\*``, ``/``, ``%`` (modulo) natively implemented::

    >>> 7 * 3.
    21.0
    >>> a = 8
    >>> b = 3
    >>> a/b # Integer division corresponds to Euclidean division
    2
    >>> float(a)/b # float() transforms a number (here an int) into a
    >>> # float
    2.6666666666666665
    >>> a%3
    2

**Strings** 

Strings are delimited by simple or double quotes::

    >>> "hello"
    'hello'
    >>> 'hello'
    'hello'
    >>> "what's up?"
    "what's up?"

.. sourcecode:: ipython

    In [9]: 'what's up'
    ------------------------------------------------------------
       File "<ipython console>", line 1
	 'what's up'
               ^
    SyntaxError: invalid syntax


As seen above, strings are concatenated with ``+`` and repeated with ``*`` ::

    >>> "how " + "are" + " you?" 
    'how are you?'
    >>> 2*"hello "
    'hello hello '

The newline character is ``\n``, and the tab characted is
``\t``.

The n*th* character of a string ``s`` is ``s[n]``::

    >>> a = "hello"
    >>> a[0]
    'h'
    >>> a[1]
    'e'
    >>> a[-1]
    'o'

Careful: **the first character of a string has index 0** (like in C), not
1 (like in Fortran or Matlab)! 

Negative indices correspond to counting from the right end.

It is also possible to define a substring of regularly spaced characters,
called a **slice**
::

    >>> a = "hello, world!"
    >>> a[3:6] # 3rd to 6th (excluded) elements: elements 3, 4, 5
    'lo,'
    >>> # the a[start:stop] slice has (strop - start) elements
    >>> a[2:10:2] # Syntax: a[start:stop:step]
    'lo o'
    >>> a[::3] # every three characters, from beginning to end 
    'hl r!'
    >>> a[:10] # the ten first characters
    'hello, wor'
    >>> a[::-1] # running backwards
    '!dlrow ,olleh'

Accents and special characters can also be handled in Unicode strings (see
http://docs.python.org/tutorial/introduction.html#unicode-strings).


A string is an immutable object and it is not possible to modify its
characters. One may however create new strings from an original one.

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

.. warning:: 

    Python offers advanced possibilities for manipulating strings,
    looking for patterns or formatting. Due to lack of time this topic is
    not addressed here, but the interested reader is referred to
    http://docs.python.org/library/stdtypes.html#string-methods and
    http://docs.python.org/library/string.html#new-string-formatting

**Lists**

.. put lists before strings? And add blurb about lists vs arrays.

A list is an ordered collection of objects, that may have different
types. For example ::

    >>> l = [3, 2, 'hello']
    >>> l
    [3, 2, 'hello']

The elements of a list are accessed by **indexing** the list as for strings.
Also, sub-lists are obtained by **slicing** ::

    >>> l[0]
    3
    >>> l[-1]
    'hello'
    >>> l[1:]
    [2, 'hello']
    >>> l[::2]
    [3, 'hello']

Unlike strings, a list is mutable and its elements can be modified::

    >>> l[0] = 1
    >>> l
    [1, 2, 'hello']

As for strings, Python offers a large panel of functions to modify lists,
or query them. Here are a few examples; for more details, see
http://docs.python.org/tutorial/datastructures.html#more-on-lists ::

    >>> a = [66.25, 333, 333, 1, 1234.5]
    >>> print a.count(333), a.count(66.25), a.count('x')
    2 1 0
    >>> a.insert(2, -1)
    >>> a.append(333)
    >>> a
    [66.25, 333, -1, 333, 1, 1234.5, 333]
    >>> a.index(333)
    1
    >>> a.remove(333)
    >>> a
    [66.25, -1, 333, 1, 1234.5, 333]
    >>> a.reverse()
    >>> a
    [333, 1234.5, 1, 333, -1, 66.25]
    >>> a.sort()
    >>> a
    [-1, 1, 66.25, 333, 333, 1234.5]

The notation ``a.function()`` is our first example of object-oriented
programming (OOP). Being a ``list``, the object `a` owns the *method*
`function` that is called using the notation **.**. No further knowledge
of OOP than understanding the notation **.** is necessary for going
through this tutorial.  

**Tuples**

Tuples are basically immutable lists. The elements of a tuple are written
between brackets, or just separated by commas::


    >>> t = 12345, 54321, 'hello!'
    >>> t[0]
    12345
    >>> t
    (12345, 54321, 'hello!')
    >>> u = (0, 2)

**Dictionnaries**

A dictionnary is basically a hash table that **maps keys to values**. It
is therefore an unordered container::


    >>> tel = {'emmanuelle': 5752, 'sebastian': 5578}
    >>> tel['francis'] = 5915 
    >>> tel
    {'sebastian': 5578, 'francis': 5915, 'emmanuelle': 5752}
    >>> tel['sebastian']
    5578
    >>> tel.keys()
    ['sebastian', 'francis', 'emmanuelle']
    >>> 'francis' in tel
    True

This is a very convenient data container in order to store values
associated to a name (a string for a date, a name, etc.). See
http://docs.python.org/tutorial/datastructures.html#dictionaries
for more information.

Flow control
------------

**Defining functions**

We now define a function that computes the ``n`` first terms of Fibonacci
sequence. Now type the following line in your Python interpreter, and be
careful to **respect the indentation depth**. The Ipython shells
automatically increases the indentation depth after a **:** sign; to
decrease the indentation depth, go four spaces to the left with the
Backspace key. Press the Enter key twice to leave the function
definition. ::

    >>> def fib(n):    
    ...     """Display the n first terms of Fibonacci sequence"""
    ...     a, b = 0, 1
    ...     i = 0
    ...     while i < n:
    ...         print b
    ...         a, b = b, a+b
    ...         i +=1
    ...
    >>> fib(10)
    1
    1 
    2
    3
    5
    8
    13
    21
    34
    55
 

Another example::

    >>> def message(name, country='France'):
    ...     message = "Hello, my name is %s and I live in %s."%(name, country)
    ...     return message # the output of the function
    ... 
    >>> message('Emma')
    'Hello, my name is Emma and I live in France.'
    >>> message('Mike', country='Germany')
    'Hello, my name is Mike and I live in Germany.'
    >>> message('Mike', 'Germany')
    'Hello, my name is Mike and I live in Germany.'


Note the syntax to define a function:

    * the ``def`` keyword;
    
    * is followed by the function's **name**, then

    * the arguments of the function are given between brackets followed
      by a colon. 

    * the function body ;

    * in order to finally return an object as output, use the syntax
      ``return object``.

Note that it is possible to define **optional arguments**, the default
value of which is set in the definition of the function. These arguments
are known as **keyword arguments**. This is a very convenient feature for
defining functions with a variable number of arguments, especially when
default values are to be used in most calls to the function.

.. warning:: 

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

    In particular, one should not start a newline in the middle of an
    instruction. Long lines can nevertheless be broken with ``\``::
   
	>>> long_line = "Here is a very very long line \
	... that we break in two parts."
 
    All this indentation business can be a bit confusing in the
    beginning. However, with the clear indentation, and in the absence of
    extra characters, the resulting code is very nice to read compared to
    other languages.


As in most languages, one can write ``for``and ``while`` loops, or test
conditions with ``if`` and ``else`` ::

    >>> # range(start, stop, step) returns a list of integers
    >>> l = range(0, 10) 
    >>> l     
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> for number in l:
    ...     if (number%2 == 0):
    ...         print number
    ...     else:
    ...         print "odd number"
    ...         
    0
    odd number
    2
    odd number
    4
    odd number
    6
    odd number
    8
    odd number

Note that ``if`` and ``else`` have the same indentation depth (use the
backspace key to decrease the indentation depth in Ipython).

It is possible to loop over other objects than integer indices. For
example, Python can loop over the elements of a list or the characters of
a string::

    >>> message = "hello"
    >>> for c in message:
    ...     print c
    ...     
    h
    e
    l
    l
    o
    >>> message = "Hello how are you?"
    >>> message.split()
    ['Hello', 'how', 'are', 'you?']
    >>> for word in message.split():
    ...     print word
    ...     
    Hello
    how
    are
    you?
    >>> l = [[1, 2, 3], 'hello', [5, 6]]
    >>> for element in l:
    ...     print element
    ...     
    [1, 2, 3]
    hello
    [5, 6]


Few languages (in particular, languages for scienfic computing) allow to
loop over anything but integers/indices. With Python it is possible to
loop exactly over the objects of interest without bothering with indices
you often don't care about.


Scripts and modules
---------------------

For now, we have typed all instructions in the interpreter. For longer
sets of instructions we need to change tack and write the code in
scripts, using a text editor. Use your favorite text editor (provided it
offers syntax highlighting for Python), or the editor that comes with the
Scientific Python Suite you may be using (e.g., Scite with Python(x,y)). 

Let us first write a **script**, that is a file with a sequence of
instructions, that may be e.g. copied-and-pasted from the interpreter
(but take care to respect indentation rules!). The extension for Python
files is **.py**. Write or copy-and-paste the following lines in a file
called **test.py** ::

    message = "Hello how are you?"
    for word in message.split():
        print word

In order to execute this script, you may

    * execute it in a console (Linux/Mac console or cmd
      Windows console). For example, if we are in the same directory as the
      test.py file, we can execute this in a console:

.. sourcecode:: bash 

    epsilon:~/sandbox$ python test.py
    Hello
    how
    are
    you?

However, this is not an interactive use, and for scientific computing we mostly
work in interactive mode, inside an interpreter:

    * in Ipython, the syntax to execute a script is ``%run
      script.py`` (don't forget the ``%`` in front of ``run`` !). For example, 

.. sourcecode:: ipython

    In [1]: %run test.py
    Hello
    how
    are
    you?

    In [2]: message
    Out[2]: 'Hello how are you?'


The script has been executed. Moreover the variables defined in the script now
are accessible (such as ``message``).

If we want to write larger and better organized programs, where some objects are defined,
(variables, functions, classes) and that we want to reuse several times, we have
to create a **module**. Below is an example of a module, contained in the
sequences.py file (copy-paste the contents in a file named sequences.py)::

    def fib(n):
        "return nth term of Fibonacci sequence"
        a, b = 0, 1
        i = 0
        while i<n:
            a, b = b, a+b
            i += 1
        return b
    
    def linear_recurrence(n, (a,b)=(2,0), (u0, u1)=(1,1)):
        """return nth term of the sequence defined by the
        linear recurrence
            u(n+2) = a*u(n+1) + b*u(n)"""
        i = 0
        u, v = u0, u1
        while i<n:
            w = a*v + b*u
            u, v = v, w
            i +=1
        return w

In this file, we defined two kinds of sequence. Suppose we want to call
the ``fib`` function from the interpreter. We could execute the module as
a script, but since there is no instructions to execute, we are rather
going to **import it as a module**. The syntax is as follows::

    >>> import sequences
    >>> sequences.linear_recurrence(10)
    1024
    >>> for i in range(5):
    ...     print i, sequences.fib(i)
    ...     
    0 1
    1 1
    2 2
    3 3
    4 5

The code in the file is executed during import of the module. Then we can use
the objects it defines, thanks to the ``module.object`` syntax. Don't forget to
put the module name before the object name, otherwise Python won't recognize the
instruction.

If we want to avoid typing ``module.object`` each time, we can import some or
all of the objects into the main namespace. For instance::

    >>>from sequences import fib
    >>> fib(10)
    89
    >>> # ou
    >>> from sequences import *
    >>> linear_recurrence(5)
    32


.. sourcecode:: ipython

    In [29]: who
    fib linear_recurrence	

    In [30]: whos
    Variable            Type        Data/Info
    -----------------------------------------
    fib                 function    <function fib at 0x96eb8ec>
    linear_recurrence   function    <function linear_recurrence at 0x96eb9cc>


When using ``from module import *``, be careful to not overwrite an already
existing object (for example, if we already had a function or a variable named
``fib``). This method should be avoided with module containing a lot of objects,
or conflicting names (max, mean, etc.).


To shorten the names, we can import a module as another name. For example, a
convention is to import ``numpy`` (which we are soon going to learn) as
``np``::

    >>> import numpy as np
    >>> type(np)
    <type 'module'>

Submodules can be defined in modules::

    >>> import scipy # routines de calcul scientifique
    >>> import scipy.optimize # sous-module d'optimisation
    >>> type(scipy.optimize)
    <type 'module'>
    >>> import scipy.optimize as opti # plus court !


Modules are thus a good way to organize code in a hierarchical way. Actually,
all the scientific computing tools we are going to use are modules::

    >>> import numpy as np # data arrays
    >>> np.linspace(0, 10, 6)
    array([  0.,   2.,   4.,   6.,   8.,  10.])
    >>> import scipy # scientific computing
    >>> from pylab import * # plotting
    >>> # calling Ipython with the -pylab switch is equivalent
    >>> # to the previous line (ipython -pylab)

As we've already seen, when we are writing a well-organized code file (ex:
``suites.py``, we are just creating a module.

In Python(x,y) software, Ipython(x,y) execute the following imports at startup::

    >>> import numpy	
    >>> import numpy as np
    >>> from pylab import *
    >>> import scipy

then we won't have to replay these imports.



Input and Output
----------------

To be exhaustive, here are some informations about input and output in Python.
Since we will use the Numpy methods to read and write files, you can skip this
chapter in first read.

We write or read **strings** to/from files (other types must be converted to
strings). To write in a file::
::

    >>> f = open('workfile', 'w') # ouvre le fichier workfile
    >>> type(f)
    <type 'file'>
    >>> f.write('Ceci est un test \nEncore un test')
    >>> f.close()

To read from a file::

    >>> f = open('workfile', 'r')
    >>> s = f.read()
    >>> print s
    Ceci est un test 
    Encore un test
    >>> f.close()

For more details: http://docs.python.org/tutorial/inputoutput.html


.. toctree::
    :maxdepth: 1

    language/exceptions.rst


.. toctree::
    :maxdepth: 1

    language/oop.rst

