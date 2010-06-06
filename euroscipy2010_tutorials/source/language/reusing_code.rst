Reusing code: scripts and modules
---------------------------------

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
suites.py file (copy-paste the contents in a file named suites.py)::

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

In this file, we defined two kinds of suite. Suppose we want to call the ``fib``
function from the interpreter. We could execute the module as a script, but
since there is no instructions to execute, we are rather going to **import it as
a module**. The syntax is as follows::

    >>> import suites
    >>> suites.linear_recurrence(10)
    1024
    >>> for i in range(5):
    ...     print i, suites.fib(i)
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

    >>>from suites import fib
    >>> fib(10)
    89
    >>> # ou
    >>> from suites import *
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




