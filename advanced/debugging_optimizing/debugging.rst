===========
 Debugging
===========

The python debugger ``pdb``: http://docs.python.org/library/pdb.html

Coding best practices to avoid getting in trouble
--------------------------------------------------

.. topic:: Brian Kernighan

   “Everyone knows that debugging is twice as hard as writing a
   program in the first place. So if you're as clever as you can be
   when you write it, how will you ever debug it?”

* We all write buggy code.  Accept it.  Deal with it.
* Write your code with testing and debugging in mind.
* Keep It Simple, Stupid (KISS).

  * What is the simplest thing that could possibly work?

* Don't Repeat Yourself (DRY).

  * Every piece of knowledge must have a single, unambiguous,
    authoritative representation within a system.
  * Constants, algorithms, etc...

* Try to limit interdependencies of your code. (Loose Coupling)
* Give your variables, functions and modules meaningful names.


The debugger
------------

A debugger allows you to inspect your code interactively.

Specifically it allows you to:

  * View the source code.
  * Walk up and down the call stack.
  * Inspect values of variables.
  * Modify values of variables.
  * Set breakpoints.

.. topic:: **print**

    Yes, ``print`` statements do work as a debugging tool. However to
    inspect runtime, it is often more efficient to use the debugger.

Invoking the debugger
^^^^^^^^^^^^^^^^^^^^^^

Ways to launch the debugger:

#. Postmortem, launch debugger after module errors.
#. Launch the module with the debugger.
#. Call the debugger inside the module


Postmortem
...........

**Situation**: You're working in ipython and you get a traceback.

Here we debug the file :download:`index_error.py`. When running it, an
`IndexError` is raised. Type ``%debug`` and drop into the debugger.

.. sourcecode:: ipython

    In [1]: %run index_error.py
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/index_error.py in <module>()
          6 
          7 if __name__ == '__main__':
    ----> 8     index_error()
          9 

    /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/index_error.py in index_error()
          3 def index_error():
          4     lst = list('foobar')
    ----> 5     print lst[len(lst)]
          6 
          7 if __name__ == '__main__':

    IndexError: list index out of range

    In [2]: %debug
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/index_error.py(5)index_error()
          4     lst = list('foobar')
    ----> 5     print lst[len(lst)]
          6 

    ipdb> list
          1 """Small snippet to raise an IndexError."""
          2 
          3 def index_error():
          4     lst = list('foobar')
    ----> 5     print lst[len(lst)]
          6 
          7 if __name__ == '__main__':
          8     index_error()
          9 

    ipdb> len(lst)
    6
    ipdb> print lst[len(lst)-1]
    r
    ipdb> quit

    In [3]: 

.. topic:: Post-mortem debugging without IPython

   In some situations you cannot use IPython, for instance to debug a
   script that wants to be called from the command line. In this case,
   you can call the script with `python -m pdb script.py`::

    $ python -m pdb index_error.py
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/index_error.py(1)<module>()
    -> """Small snippet to raise an IndexError."""
    (Pdb) continue
    Traceback (most recent call last):
    File "/usr/lib/python2.6/pdb.py", line 1296, in main
        pdb._runscript(mainpyfile)
    File "/usr/lib/python2.6/pdb.py", line 1215, in _runscript
        self.run(statement)
    File "/usr/lib/python2.6/bdb.py", line 372, in run
        exec cmd in globals, locals
    File "<string>", line 1, in <module>
    File "index_error.py", line 8, in <module>
        index_error()
    File "index_error.py", line 5, in index_error
        print lst[len(lst)]
    IndexError: list index out of range
    Uncaught exception. Entering post mortem debugging
    Running 'cont' or 'step' will restart the program
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/index_error.py(5)index_error()
    -> print lst[len(lst)]
    (Pdb) 
 


Step-by-step execution
.......................

**Situation**: You believe a bug exists in a module but are not sure where.

For instance we are trying to debug :download:`wiener_filtering.py`.
Indeed the code runs, but the filtering does not work well.

* Run the script with the debugger:

  .. sourcecode:: ipython

    In [1]: %run -d wiener_filtering.py
    *** Blank or comment
    *** Blank or comment
    *** Blank or comment
    Breakpoint 1 at /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py:4
    NOTE: Enter 'c' at the ipdb>  prompt to start your script.
    > <string>(1)<module>()

* Enter the :download:`wiener_filtering.py` file and set a break point at line
  34:

  .. sourcecode:: ipython

    ipdb> n
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py(4)<module>()
          3 
    1---> 4 import numpy as np
          5 import scipy as sp

    ipdb> b 34
    Breakpoint 2 at /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py:34

* Continue execution to next breakpoint with ``c(ont(inue))``:

  .. sourcecode:: ipython

    ipdb> c
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py(34)iterated_wiener()
         33     """
    2--> 34     noisy_img = noisy_img
         35     denoised_img = local_mean(noisy_img, size=size)

* Step into code with ``n(ext)`` and ``s(tep)``: ``next`` jumps to the next
  statement in the current execution context, while ``step`` will go across
  execution contexts, i.e. enable exploring inside function calls:

  .. sourcecode:: ipython

    ipdb> s
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py(35)iterated_wiener()
    2    34     noisy_img = noisy_img
    ---> 35     denoised_img = local_mean(noisy_img, size=size)
         36     l_var = local_var(noisy_img, size=size)

    ipdb> n
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py(36)iterated_wiener()
         35     denoised_img = local_mean(noisy_img, size=size)
    ---> 36     l_var = local_var(noisy_img, size=size)
         37     for i in range(3):


* Step a few lines and explore the local variables:

  .. sourcecode:: ipython

    ipdb> n
    > /home/varoquau/dev/scipy-lecture-notes/advanced/debugging_optimizing/wiener_filtering.py(37)iterated_wiener()
         36     l_var = local_var(noisy_img, size=size)
    ---> 37     for i in range(3):
         38         res = noisy_img - denoised_img
    ipdb> print l_var
    [[5868 5379 5316 ..., 5071 4799 5149]
     [5013  363  437 ...,  346  262 4355]
     [5379  410  344 ...,  392  604 3377]
     ..., 
     [ 435  362  308 ...,  275  198 1632]
     [ 548  392  290 ...,  248  263 1653]
     [ 466  789  736 ..., 1835 1725 1940]]
    ipdb> print l_var.min()
    0

Oh dear, nothing but integers, and 0 variation. Here is our bug, we are
doing integer arythmetic.

.. topic:: Raising exception on numerical errors

    When we run the :download:`wiener_filtering.py` file, the following
    warnings are raised:

    .. sourcecode:: ipython

        In [2]: %run wiener_filtering.py
        Warning: divide by zero encountered in divide
        Warning: divide by zero encountered in divide
        Warning: divide by zero encountered in divide

    We can turn these warnings in exception, which enables us to do
    post-mortem debugging on them, and find our problem more quickly:

    .. sourcecode:: ipython

        In [3]: np.seterr(all='raise')
        Out[3]: {'divide': 'print', 'invalid': 'print', 'over': 'print', 'under': 'ignore'}

Other ways of starting a debugger
....................................

* **Raising an exception as a poor man break point**

  If you find it tedious to note the line number to set a break point,
  you can simply raise an exception at the point that you want to
  inspect and use ipython's `%debug`. Note that in this case you cannot
  step or continue the execution.

* **Debugging test failures using nosetests**

  You can run `nosetests --pdb` to drop in post-mortem debugging on
  exceptions, and `nosetests --pdb-failure` to inspect test failures
  using the debugger.

* **Calling the debugger explicitely**

  Insert the following line where you want to drop in the debugger::

    import pdb; pdb.set_trace()

.. warning::

    When running `nosetests`, the output is captured, and thus it seems
    that the debugger does not work. Simply run the nosetests with the `-s`
    flag.


.. topic:: Graphical debuggers

    For stepping through code and inspecting variables, you might find it
    more convenient to use a graphical debugger such as 
    `winpdb <http://winpdb.org/>`_.

Debugger commands and interaction 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============ ======================================================================
``l(list)``   Lists the code at the current position
``u(p)``      Walk up the call stack
``d(own)``    Walk down the call stack
``n(ext)``    Execute the next line (does not go down in new functions)
``s(tep)``    Execute the next statement (goes down in new functions)
``bt``        Print the call stack
``a``         Print the local variables
``!command``  Exectute the given **Python** command (by opposition to pdb commands
============ ======================================================================

.. warning:: **Debugger commands are not Python code**

    You cannot name the variables the way you want. For instance, if in
    you cannot override the variables in the current frame with the same
    name: **use different names then your local variable when typing code
    in the debugger**.

Debugging strategies
--------------------

There is no silver bullet. Yet, strategies help.

   **For debugging a given problem, the favorable situation is when the
   problem is isolated in a small number of lines of code, outside
   framework or application code, with short modify-run-fail cycles**

1. Make it fail reliably.  Find a test case that makes the code fail
   every time.
2. Divide and Conquer.  Once you have a failing test case, isolate the
   failing code.

   * Which module.
   * Which function.
   * Which line of code.

   => isolate a small reproducible failure: a test case

3. Change one thing at a time and re-run the failing test case.
4. Use the debugger to inderstand what is going wrong. For instance purposely 
   raise an exception where you believe the problem is, to
   inspect the code via the debuger (eg '%debug' in IPython)
5. Take notes and be patient.  It may take a while.

____

.. topic:: **Wrap up excercise**
    
    The following script is well documented and hopefully legible. It
    seeks to answer a problem of actual interest for numerical computing,
    but it does not work... Can you debug it?

    **Python source code:** :download:`to_debug.py <to_debug.py>`

    .. literalinclude:: to_debug.py

