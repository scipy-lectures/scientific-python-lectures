First steps
-------------

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
 


