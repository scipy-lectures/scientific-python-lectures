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

Launching the debugger
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

Launch the module with the debugger and step through the code in the
debugger.

.. sourcecode:: ipython

    In [38]: run -d debug_file.py
    *** Blank or comment
    *** Blank or comment
    Breakpoint 1 at /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py:3
    NOTE: Enter 'c' at the ipdb>  prompt to start your script.
    > <string>(1)<module>()

Step into code with ``s(tep)``:

.. sourcecode:: ipython

    ipdb> step
    --Call--
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(4)<module>()
    1     3 Data is stored in data.txt.
    ----> 4 """
          5 

Set a breakpoint at the ``load_data`` function:

.. sourcecode:: ipython

    ipdb> break load_data
    Breakpoint 2 at /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py:12
    ipdb> break
    Num Type         Disp Enb   Where
    1   breakpoint   keep yes   at /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py:3
    2   breakpoint   keep yes   at /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py:12

Continue execution to next breakpoint with ``c(ont(inue))``:

.. sourcecode:: ipython

    ipdb> continue
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(13)load_data()
    2    12 def load_data(filename):
    ---> 13     fp = open(filename)
         14     data_string = fp.read()

I don't want to debug python's ``open`` function, so use the
``n(ext)`` command to continue execution on the next line:

.. sourcecode:: ipython

    ipdb> next
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(14)load_data()
         13     fp = open(filename)
    ---> 14     data_string = fp.read()
         15     fp.close()

    ipdb> next
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(15)load_data()
         14     data_string = fp.read()
    ---> 15     fp.close()
         16     return parse_data(data_string)

    ipdb> next
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(16)load_data()
         15     fp.close()
    ---> 16     return parse_data(data_string)
         17 

Step into ``parse_data`` function with ``s(tep)`` command:

.. sourcecode:: ipython

    ipdb> step
    --Call--
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(6)parse_data()
          5 
    ----> 6 def parse_data(data_string):
          7     data = []

    ipdb> list
          1 """Script to read in a column of numbers and calculate the min, max and sum.
          2 
    1     3 Data is stored in data.txt.
          4 """
          5 
    ----> 6 def parse_data(data_string):
          7     data = []
          8     for x in data_string.split('.'):
          9         data.append(x)
         10     return data
         11 

Continue stepping through code and print out values with the
``p(rint)`` command:

.. sourcecode:: ipython

    ipdb> step
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(9)parse_data()
          8     for x in data_string.split('.'):
    ----> 9         data.append(x)
         10     return data

    ipdb> p x
    '10'
    ipdb> s
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(8)parse_data()
          7     data = []
    ----> 8     for x in data_string.split('.'):
          9         data.append(x)

    ipdb> s
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(9)parse_data()
          8     for x in data_string.split('.'):
    ----> 9         data.append(x)
         10     return data

    ipdb> p x
    '2\n43'

Calling the debugger inside a module
.....................................

Insert the following line where you want to drop in the debugger::

    import pdb; pdb.set_trace()

.. warning::

    When running `nosetests`, the output is captured, and thus it seems
    that the debugger does not work. Simply run the nosetests with the `-s`
    flag.


Debugger commands and interaction 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* List the code with ``l(ist)``:

  .. sourcecode:: ipython

    ipdb> list
          1 """Script to read in a column of numbers and calculate the min, max and sum.
          2 
    1     3 Data is stored in data.txt.
    ----> 4 """
          5 
          6 def parse_data(data_string):
          7     data = []
          8     for x in data_string.split('.'):
          9         data.append(x)
         10     return data
         11 

    ipdb> list
    2    12 def load_data(filename):
         13     fp = open(filename)
         14     data_string = fp.read()
         15     fp.close()
         16     return parse_data(data_string)
         17 
         18 if __name__ == '__main__':
         19     data = load_data('exercises/data.txt')
         20     print('min: %f' % min(data)) # 10.20
         21     print('max: %f' % max(data)) # 61.30


* You can also walk up and down the call stack with ``u(p)`` and ``d(own)``:

  .. sourcecode:: ipython

    ipdb> list
          4 """
          5 
          6 def parse_data(data_string):
          7     data = []
          8     for x in data_string.split('.'):
    ----> 9         data.append(x)
         10     return data
         11 
    2    12 def load_data(filename):
         13     fp = open(filename)
         14     data_string = fp.read()

    ipdb> up
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(16)load_data()
         15     fp.close()
    ---> 16     return parse_data(data_string)
         17 

    ipdb> list
         11 
    2    12 def load_data(filename):
         13     fp = open(filename)
         14     data_string = fp.read()
         15     fp.close()
    ---> 16     return parse_data(data_string)
         17 
         18 if __name__ == '__main__':
         19     data = load_data('exercises/data.txt')
         20     print('min: %f' % min(data)) # 10.20
         21     print('max: %f' % max(data)) # 61.30

    ipdb> down
    > /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/debug_file.py(9)parse_data()
          8     for x in data_string.split('.'):
    ----> 9         data.append(x)
         10     return data

    ipdb> list
          4 """
          5 
          6 def parse_data(data_string):
          7     data = []
          8     for x in data_string.split('.'):
    ----> 9         data.append(x)
         10     return data
         11 
    2    12 def load_data(filename):
         13     fp = open(filename)
         14     data_string = fp.read()

    ipdb> 


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


