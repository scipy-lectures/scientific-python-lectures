Standard Library
================

.. note:: Reference document for this section:

 * The Python Standard Library documentation:
   https://docs.python.org/3/library/index.html

 * Python Essential Reference, David Beazley, Addison-Wesley Professional

``os`` module: operating system functionality
-----------------------------------------------

*"A portable way of using operating system dependent functionality."*

Directory and file manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current directory:

.. ipython::

    In [1]: import os

    In [2]: os.getcwd()
    Out[2]: '/home/jarrod/src/scientific-python-lectures/intro'

List a directory:

.. ipython::

    In [3]: os.listdir(os.curdir)
    Out[3]: ['intro.rst', 'scipy', 'language', 'matplotlib', 'index.rst', 'numpy', 'help']

Make a directory:

.. ipython::

    In [4]: os.mkdir('junkdir')

    In [5]: 'junkdir' in os.listdir(os.curdir)
    Out[5]: True

Rename the directory:

.. ipython::

    In [6]: os.rename('junkdir', 'foodir')

    In [7]: 'junkdir' in os.listdir(os.curdir)
    Out[7]: False

    In [8]: 'foodir' in os.listdir(os.curdir)
    Out[8]: True

    In [9]: os.rmdir('foodir')

    In [10]: 'foodir' in os.listdir(os.curdir)
    Out[10]: False

Delete a file:

.. ipython::

    In [11]: fp = open('junk.txt', 'w')

    In [12]: fp.close()

    In [13]: 'junk.txt' in os.listdir(os.curdir)
    Out[13]: True

    In [14]: os.remove('junk.txt')

    In [15]: 'junk.txt' in os.listdir(os.curdir)
    Out[15]: False

``os.path``: path manipulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``os.path`` provides common operations on pathnames.

.. ipython::

    In [16]: fp = open('junk.txt', 'w')

    In [17]: fp.close()

    In [18]: a = os.path.abspath('junk.txt')

    In [19]: a
    Out[19]: '/home/jarrod/src/scientific-python-lectures/intro/junk.txt'

    In [20]: os.path.split(a)
    Out[20]: ('/home/jarrod/src/scientific-python-lectures/intro', 'junk.txt')

    In [21]: os.path.dirname(a)
    Out[21]: '/home/jarrod/src/scientific-python-lectures/intro'

    In [22]: os.path.basename(a)
    Out[22]: 'junk.txt'

    In [23]: os.path.splitext(os.path.basename(a))
    Out[23]: ('junk', '.txt')

    In [24]: os.path.exists('junk.txt')
    Out[24]: True

    In [25]: os.path.isfile('junk.txt')
    Out[25]: True

    In [26]: os.path.isdir('junk.txt')
    Out[26]: False

    In [27]: os.path.expanduser('~/local')
    Out[27]: '/home/jarrod/local'

    In [28]: os.path.join(os.path.expanduser('~'), 'local', 'bin')
    Out[28]: '/home/jarrod/local/bin'

Running an external command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ipython::

    In [29]: os.system('ls')
    help  index.rst  intro.rst  junk.txt  language	matplotlib  numpy  scipy
    Out[29]: 0

.. note:: Alternative to ``os.system``

    A noteworthy alternative to ``os.system`` is the `sh module
    <https://amoffat.github.com/sh/>`_. Which provides much more convenient ways to
    obtain the output, error stream and exit code of the external command.

    .. ipython::
        :verbatim:

        In [30]: import sh
        In [31]: com = sh.ls()

        In [32]: print(com)
        basic_types.rst   exceptions.rst   oop.rst              standard_library.rst
        control_flow.rst  first_steps.rst  python_language.rst
        demo2.py          functions.rst    python-logo.png
        demo.py           io.rst           reusing_code.rst

        In [33]: type(com)
        Out[33]: str

Walking a directory
~~~~~~~~~~~~~~~~~~~~

``os.path.walk`` generates a list of filenames in a directory tree.

.. ipython::

    In [10]: for dirpath, dirnames, filenames in os.walk(os.curdir):
       ....:     for fp in filenames:
       ....:         print(os.path.abspath(fp))
       ....:
       ....:
    /home/jarrod/src/scientific-python-lectures/intro/language/basic_types.rst
    /home/jarrod/src/scientific-python-lectures/intro/language/control_flow.rst
    /home/jarrod/src/scientific-python-lectures/intro/language/python_language.rst
    /home/jarrod/src/scientific-python-lectures/intro/language/reusing_code.rst
    /home/jarrod/src/scientific-python-lectures/intro/language/standard_library.rst
    ...

Environment variables:
~~~~~~~~~~~~~~~~~~~~~~

.. ipython::
    :verbatim:

    In [32]: os.environ.keys()
    Out[32]: KeysView(environ({'SHELL': '/bin/bash', 'COLORTERM': 'truecolor', ...}))


    In [34]: os.environ['SHELL']
    Out[34]: '/bin/bash'


``shutil``: high-level file operations
---------------------------------------

The ``shutil`` provides useful file operations:

    * ``shutil.rmtree``: Recursively delete a directory tree.
    * ``shutil.move``: Recursively move a file or directory to another location.
    * ``shutil.copy``: Copy files or directories.

``glob``: Pattern matching on files
-------------------------------------

The ``glob`` module provides convenient file pattern matching.

Find all files ending in ``.txt``:

.. ipython::

    In [36]: import glob

    In [37]: glob.glob('*.txt')
    Out[37]: ['junk.txt']

``sys`` module: system-specific information
--------------------------------------------

System-specific information related to the Python interpreter.

* Which version of python are you running and where is it installed:

.. ipython::


    In [39]: import sys

    In [40]: sys.platform
    Out[40]: 'linux'

    In [41]: sys.version
    Out[41]: '3.11.8 (main, Feb 28 2024, 00:00:00) [GCC 13.2.1 20231011 (Red Hat 13.2.1-4)]'

    In [42]: sys.prefix
    Out[42]: '/home/jarrod/.venv/nx'

* List of command line arguments passed to a Python script:

.. ipython::

    In [43]: sys.argv
    Out[43]: ['/home/jarrod/.venv/nx/bin/ipython']

``sys.path`` is a list of strings that specifies the search path for
modules.  Initialized from PYTHONPATH:

.. ipython::

    In [44]: sys.path
    Out[44]:
    ['/home/jarrod/.venv/nx/bin',
     '/usr/lib64/python311.zip',
     '/usr/lib64/python3.11',
     '/usr/lib64/python3.11/lib-dynload',
     '',
     '/home/jarrod/.venv/nx/lib64/python3.11/site-packages',
     '/home/jarrod/.venv/nx/lib/python3.11/site-packages']

``pickle``: easy persistence
-------------------------------

Useful to store arbitrary objects to a file. Not safe or fast!

.. ipython::

    In [45]: import pickle

    In [46]: l = [1, None, 'Stan']

    In [3]: with open('test.pkl', 'wb') as file:
       ...:     pickle.dump(l, file)
       ...:

    In [4]: with open('test.pkl', 'rb') as file:
       ...:     out = pickle.load(file)
       ...:

    In [49]: out
    Out[49]: [1, None, 'Stan']


.. topic:: Exercise

    Write a program to search your ``PYTHONPATH`` for the module ``site.py``.

:ref:`path_site`
