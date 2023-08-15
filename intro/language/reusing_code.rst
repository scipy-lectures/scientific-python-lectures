Reusing code: scripts and modules
=================================

For now, we have typed all instructions in the interpreter. For longer
sets of instructions we need to change track and write the code in text
files (using a text editor), that we will call either *scripts* or
*modules*. Use your favorite text editor (provided it offers syntax
highlighting for Python), or the editor that comes with the Scientific
Python Suite you may be using.

Scripts
-------

.. tip::

    Let us first write a *script*, that is a file with a sequence of
    instructions that are executed each time the script is called.
    Instructions may be e.g. copied-and-pasted from the interpreter (but
    take care to respect indentation rules!).

The extension for Python files is ``.py``. Write or copy-and-paste the
following lines in a file called ``test.py`` ::

    message = "Hello how are you?"
    for word in message.split():
        print(word)

.. tip::

    Let us now execute the script interactively, that is inside the
    Ipython interpreter. This is maybe the most common use of scripts in
    scientific computing.

.. note::

    in Ipython, the syntax to execute a script is ``%run script.py``. For
    example,

.. ipython::
    :verbatim:

    In [1]: %run test.py
    Hello
    how
    are
    you?

    In [2]: message
    Out[2]: 'Hello how are you?'


The script has been executed. Moreover the variables defined in the
script (such as ``message``) are now available inside the interpreter's
namespace.

.. tip::

    Other interpreters also offer the possibility to execute scripts
    (e.g., ``execfile`` in the plain Python interpreter, etc.).

It is also possible In order to execute this script as a *standalone
program*, by executing the script inside a shell terminal (Linux/Mac
console or cmd Windows console). For example, if we are in the same
directory as the test.py file, we can execute this in a console:

.. sourcecode:: bash

    $ python test.py
    Hello
    how
    are
    you?

.. tip::

    Standalone scripts may also take command-line arguments

    In ``file.py``::

        import sys
        print(sys.argv)

    .. sourcecode:: bash

        $ python file.py test arguments
        ['file.py', 'test', 'arguments']

    .. warning::

        Don't implement option parsing yourself. Use a dedicated module such as
        :mod:`argparse`.


Importing objects from modules
------------------------------

.. ipython::

    In [1]: import os

    In [2]: os
    Out[2]: <module 'os' from '/usr/lib64/python3.10/os.py'>

    In [3]: os.listdir('.')
    Out[3]:
    ['conf.py',
     'basic_types.rst',
     'control_flow.rst',
     'functions.rst',
     'python_language.rst',
     'reusing.rst',
     'file_io.rst',
     'exceptions.rst',
     'workflow.rst',
     'index.rst']

And also:

.. ipython::

    In [4]: from os import listdir

Importing shorthands:

.. ipython::

    In [5]: import numpy as np

.. warning::

    ::

        from os import *

    This is called the *star import* and please, **Do not use it**

    * Makes the code harder to read and understand: where do symbols come
      from?

    * Makes it impossible to guess the functionality by the context and
      the name (hint: `os.name` is the name of the OS), and to profit
      usefully from tab completion.

    * Restricts the variable names you can use: `os.name` might override
      `name`, or vise-versa.

    * Creates possible name clashes between modules.

    * Makes the code impossible to statically check for undefined
      symbols.

.. tip::

  Modules are thus a good way to organize code in a hierarchical way. Actually,
  all the scientific computing tools we are going to use are modules::

    >>> import numpy as np # data arrays
    >>> np.linspace(0, 10, 6)
    array([  0.,   2.,   4.,   6.,   8.,  10.])
    >>> import scipy as sp  # scientific computing


Creating modules
-----------------

.. tip::

    If we want to write larger and better organized programs (compared to
    simple scripts), where some objects are defined, (variables,
    functions, classes) and that we want to reuse several times, we have
    to create our own *modules*.

Let us create a module ``demo`` contained in the file ``demo.py``:

  .. literalinclude:: demo.py

.. tip::

    In this file, we defined two functions ``print_a`` and ``print_b``. Suppose
    we want to call the ``print_a`` function from the interpreter. We could
    execute the file as a script, but since we just want to have access to
    the function ``print_a``, we are rather going to **import it as a module**.
    The syntax is as follows.


.. ipython::
    :verbatim:

    In [1]: import demo


    In [2]: demo.print_a()
    a

    In [3]: demo.print_b()
    b

Importing the module gives access to its objects, using the
``module.object`` syntax. Don't forget to put the module's name before the
object's name, otherwise Python won't recognize the instruction.


Introspection

.. ipython::
    :verbatim:

    In [4]: demo?
    Type:               module
    Base Class: <type 'module'>
    String Form:        <module 'demo' from 'demo.py'>
    Namespace:  Interactive
    File:               /home/varoquau/Projects/Python_talks/scipy_2009_tutorial/source/demo.py
    Docstring:
        A demo module.


    In [5]: who
    demo

    In [6]: whos
    Variable   Type      Data/Info
    ------------------------------
    demo       module    <module 'demo' from 'demo.py'>

    In [7]: dir(demo)
    Out[7]:
    ['__builtins__',
    '__doc__',
    '__file__',
    '__name__',
    '__package__',
    'c',
    'd',
    'print_a',
    'print_b']


    In [8]: demo.<TAB>
    demo.c        demo.print_a  demo.py
    demo.d        demo.print_b  demo.pyc


Importing objects from modules into the main namespace

.. ipython::
    :verbatim:

    In [9]: from demo import print_a, print_b

    In [10]: whos
    Variable   Type        Data/Info
    --------------------------------
    demo       module      <module 'demo' from 'demo.py'>
    print_a    function    <function print_a at 0xb7421534>
    print_b    function    <function print_b at 0xb74214c4>

    In [11]: print_a()
    a

.. warning::

    **Module caching**

     Modules are cached: if you modify ``demo.py`` and re-import it in the
     old session, you will get the old one.

    Solution:

     .. sourcecode :: ipython

        In [10]: importlib.reload(demo)

'__main__' and module loading
------------------------------

.. tip::

    Sometimes we want code to be executed when a module is
    run directly, but not when it is imported by another module.
    ``if __name__ == '__main__'`` allows us to check whether the
    module is being run directly.

File ``demo2.py``:

  .. literalinclude:: demo2.py

Importing it:

.. ipython::
    :verbatim:

    In [11]: import demo2
    b

    In [12]: import demo2

Running it:

.. ipython::
    :verbatim:

    In [13]: %run demo2
    b
    a


Scripts or modules? How to organize your code
---------------------------------------------

.. Note:: Rule of thumb

    * Sets of instructions that are called several times should be
      written inside **functions** for better code reusability.

    * Functions (or other bits of code) that are called from several
      scripts should be written inside a **module**, so that only the
      module is imported in the different scripts (do not copy-and-paste
      your functions in the different scripts!).

How modules are found and imported
..................................


When the ``import mymodule`` statement is executed, the module ``mymodule``
is searched in a given list of directories. This list includes a list
of installation-dependent default path (e.g., ``/usr/lib64/python3.11``) as
well as the list of directories specified by the environment variable
``PYTHONPATH``.

The list of directories searched by Python is given by the ``sys.path``
variable

.. ipython::

    In [1]: import sys

    In [2]: sys.path
    Out[2]:
    ['/home/jarrod/.venv/lectures/bin',
     '/usr/lib64/python311.zip',
     '/usr/lib64/python3.11',
     '/usr/lib64/python3.11/lib-dynload',
     '',
     '/home/jarrod/.venv/lectures/lib64/python3.11/site-packages',
     '/home/jarrod/.venv/lectures/lib/python3.11/site-packages']

Modules must be located in the search path, therefore you can:

* write your own modules within directories already defined in the
  search path (e.g. ``$HOME/.venv/lectures/lib64/python3.11/site-packages``).
  You may use symbolic links (on Linux) to keep the code somewhere else.

* modify the environment variable ``PYTHONPATH`` to include the
  directories containing the user-defined modules.

  .. tip::

    On Linux/Unix, add the following line to a file read by the shell at
    startup (e.g. /etc/profile, .profile)

    ::

      export PYTHONPATH=$PYTHONPATH:/home/emma/user_defined_modules

    On Windows, https://support.microsoft.com/kb/310519 explains how to
    handle environment variables.

* or modify the ``sys.path`` variable itself within a Python script.

  .. tip::

    ::

        import sys
        new_path = '/home/emma/user_defined_modules'
        if new_path not in sys.path:
            sys.path.append(new_path)

    This method is not very robust, however, because it makes the code
    less portable (user-dependent path) and because you have to add the
    directory to your sys.path each time you want to import from a module
    in this directory.

.. seealso::

    See https://docs.python.org/3/tutorial/modules.html for more information
    about modules.

Packages
--------

A directory that contains many modules is called a *package*. A package
is a module with submodules (which can have submodules themselves, etc.).
A special file called ``__init__.py`` (which may be empty) tells Python
that the directory is a Python package, from which modules can be
imported.

.. sourcecode:: bash

    $ ls
    _build_utils/         fft/          _lib/        odr/          spatial/
    cluster/              fftpack/      linalg/      optimize/     special/
    conftest.py           __init__.py   linalg.pxd   optimize.pxd  special.pxd
    constants/            integrate/    meson.build  setup.py      stats/
    datasets/             interpolate/  misc/        signal/
    _distributor_init.py  io/           ndimage/     sparse/
    $ cd ndimage
    $ ls
    _filters.py  __init__.py        _measurements.py  morphology.py      src/
    filters.py   _interpolation.py  measurements.py   _ni_docstrings.py  tests/
    _fourier.py  interpolation.py   meson.build       _ni_support.py     utils/
    fourier.py   LICENSE.txt        _morphology.py    setup.py


From Ipython:

.. ipython::

    In [1]: import scipy as sp

    In [2]: sp.__file__

    In [3]: sp.version.version

    @verbatim
    In [4]: sp.ndimage.morphology.binary_dilation?
    Signature:
    sp.ndimage.morphology.binary_dilation(
        input,
        structure=None,
        iterations=1,
        mask=None,
        output=None,
        border_value=0,
        origin=0,
        brute_force=False,
    )
    Docstring:
    Multidimensional binary dilation with the given structuring element.
    ...


Good practices
--------------

* Use **meaningful** object **names**

* **Indentation: no choice!**

  .. tip::

    Indenting is compulsory in Python! Every command block following a
    colon bears an additional indentation level with respect to the
    previous line with a colon. One must therefore indent after
    ``def f():`` or ``while:``. At the end of such logical blocks, one
    decreases the indentation depth (and re-increases it if a new block
    is entered, etc.)

    Strict respect of indentation is the price to pay for getting rid of
    ``{`` or ``;`` characters that delineate logical blocks in other
    languages. Improper indentation leads to errors such as

    .. code-block:: ipython

        ------------------------------------------------------------
        IndentationError: unexpected indent (test.py, line 2)

    All this indentation business can be a bit confusing in the
    beginning. However, with the clear indentation, and in the absence of
    extra characters, the resulting code is very nice to read compared to
    other languages.

* **Indentation depth**: Inside your text editor, you may choose to
  indent with any positive number of spaces (1, 2, 3, 4, ...). However,
  it is considered good practice to **indent with 4 spaces**. You may
  configure your editor to map the ``Tab`` key to a 4-space
  indentation.

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
  Code <https://peps.python.org/pep-0008>`_.


____


.. topic:: **Quick read**

   If you want to do a first quick pass through the Scientific Python Lectures
   to learn the ecosystem, you can directly skip to the next chapter:
   :ref:`numpy`.

   The remainder of this chapter is not necessary to follow the rest of
   the intro part. But be sure to come back and finish this chapter later.
