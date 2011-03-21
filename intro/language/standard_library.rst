Standard Library
================

.. note:: Reference document for this section:

 * The Python Standard Library documentation: 
   http://docs.python.org/library/index.html

 * Python Essential Reference, David Beazley, Addison-Wesley Professional

``os`` module: operating system functionality
-----------------------------------------------

*"A portable way of using operating system dependent functionality."*

Directory and file manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current directory:

.. sourcecode:: ipython

    In [17]: os.getcwd()
    Out[17]: '/Users/cburns/src/scipy2009/scipy_2009_tutorial/source'

List a directory:

.. sourcecode:: ipython

    In [31]: os.listdir(os.curdir)
    Out[31]: 
    ['.index.rst.swo',
     '.python_language.rst.swp',
     '.view_array.py.swp',
     '_static',
     '_templates',
     'basic_types.rst',
     'conf.py',
     'control_flow.rst',
     'debugging.rst',
     ...

Make a directory:

.. sourcecode:: ipython

    In [32]: os.mkdir('junkdir')

    In [33]: 'junkdir' in os.listdir(os.curdir)
    Out[33]: True

Rename the directory:

.. sourcecode:: ipython

    In [36]: os.rename('junkdir', 'foodir')

    In [37]: 'junkdir' in os.listdir(os.curdir)
    Out[37]: False

    In [38]: 'foodir' in os.listdir(os.curdir)
    Out[38]: True

    In [41]: os.rmdir('foodir')

    In [42]: 'foodir' in os.listdir(os.curdir)
    Out[42]: False

Delete a file:

.. sourcecode:: ipython

    In [44]: fp = open('junk.txt', 'w')

    In [45]: fp.close()

    In [46]: 'junk.txt' in os.listdir(os.curdir)
    Out[46]: True

    In [47]: os.remove('junk.txt')

    In [48]: 'junk.txt' in os.listdir(os.curdir)
    Out[48]: False

``os.path``: path manipulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``os.path`` provides common operations on pathnames.

.. sourcecode:: ipython

    In [70]: fp = open('junk.txt', 'w')

    In [71]: fp.close()

    In [72]: a = os.path.abspath('junk.txt')

    In [73]: a
    Out[73]: '/Users/cburns/src/scipy2009/scipy_2009_tutorial/source/junk.txt'

    In [74]: os.path.split(a)
    Out[74]: ('/Users/cburns/src/scipy2009/scipy_2009_tutorial/source', 
              'junk.txt')

    In [78]: os.path.dirname(a)
    Out[78]: '/Users/cburns/src/scipy2009/scipy_2009_tutorial/source'

    In [79]: os.path.basename(a)
    Out[79]: 'junk.txt'

    In [80]: os.path.splitext(os.path.basename(a))
    Out[80]: ('junk', '.txt')

    In [84]: os.path.exists('junk.txt')
    Out[84]: True

    In [86]: os.path.isfile('junk.txt')
    Out[86]: True

    In [87]: os.path.isdir('junk.txt')
    Out[87]: False

    In [88]: os.path.expanduser('~/local')
    Out[88]: '/Users/cburns/local'

    In [92]: os.path.join(os.path.expanduser('~'), 'local', 'bin')
    Out[92]: '/Users/cburns/local/bin'

Running an external command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. sourcecode:: ipython

  In [8]: os.system('ls *')
  conf.py   debug_file.py  demo2.py~  demo.py   demo.pyc	  my_file.py~
  conf.py~  demo2.py	 demo2.pyc  demo.py~  my_file.py  pi_wallis_image.py


Walking a directory
~~~~~~~~~~~~~~~~~~~~

``os.path.walk`` generates a list of filenames in a directory tree.

.. sourcecode:: ipython

    In [10]: for dirpath, dirnames, filenames in os.walk(os.curdir):
       ....:     for fp in filenames:
       ....:         print os.path.abspath(fp)
       ....:         
       ....:         
    /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/.index.rst.swo
    /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/.view_array.py.swp
    /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/basic_types.rst
    /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/conf.py
    /Users/cburns/src/scipy2009/scipy_2009_tutorial/source/control_flow.rst
    ...

Environment variables:
~~~~~~~~~~~~~~~~~~~~~~

.. sourcecode:: ipython

    In [9]: import os

    In [11]: os.environ.keys()
    Out[11]: 
    ['_',
     'FSLDIR',
     'TERM_PROGRAM_VERSION',
     'FSLREMOTECALL',
     'USER',
     'HOME',
     'PATH',
     'PS1',
     'SHELL',
     'EDITOR',
     'WORKON_HOME',
     'PYTHONPATH',
     ...

    In [12]: os.environ['PYTHONPATH']
    Out[12]: '.:/Users/cburns/src/utils:/Users/cburns/src/nitools:
    /Users/cburns/local/lib/python2.5/site-packages/:
    /usr/local/lib/python2.5/site-packages/:
    /Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5'

    In [16]: os.getenv('PYTHONPATH')
    Out[16]: '.:/Users/cburns/src/utils:/Users/cburns/src/nitools:
    /Users/cburns/local/lib/python2.5/site-packages/:
    /usr/local/lib/python2.5/site-packages/:
    /Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5'


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

.. sourcecode:: ipython

    In [18]: import glob

    In [19]: glob.glob('*.txt')
    Out[19]: ['holy_grail.txt', 'junk.txt', 'newfile.txt']


``sys`` module: system-specific information
--------------------------------------------

System-specific information related to the Python interpreter.

* Which version of python are you running and where is it installed:

  .. sourcecode:: ipython

    In [117]: sys.platform
    Out[117]: 'darwin'

    In [118]: sys.version
    Out[118]: '2.5.2 (r252:60911, Feb 22 2008, 07:57:53) \n
              [GCC 4.0.1 (Apple Computer, Inc. build 5363)]'

    In [119]: sys.prefix
    Out[119]: '/Library/Frameworks/Python.framework/Versions/2.5'

* List of command line arguments passed to a Python script:

  .. sourcecode:: ipython

   In [100]: sys.argv
   Out[100]: ['/Users/cburns/local/bin/ipython']


``sys.path`` is a list of strings that specifies the search path for
modules.  Initialized from PYTHONPATH:

.. sourcecode:: ipython

    In [121]: sys.path
    Out[121]: 
    ['',
     '/Users/cburns/local/bin',
     '/Users/cburns/local/lib/python2.5/site-packages/grin-1.1-py2.5.egg',
     '/Users/cburns/local/lib/python2.5/site-packages/argparse-0.8.0-py2.5.egg',
     '/Users/cburns/local/lib/python2.5/site-packages/urwid-0.9.7.1-py2.5.egg',
     '/Users/cburns/local/lib/python2.5/site-packages/yolk-0.4.1-py2.5.egg',
     '/Users/cburns/local/lib/python2.5/site-packages/virtualenv-1.2-py2.5.egg',
     ...

``pickle``: easy persistence
-------------------------------

Useful to store arbitrary objects to a file. Not safe or fast!

.. sourcecode:: ipython

  In [1]: import pickle

  In [2]: l = [1, None, 'Stan']

  In [3]: pickle.dump(l, file('test.pkl', 'w'))

  In [4]: pickle.load(file('test.pkl'))
  Out[4]: [1, None, 'Stan']


.. topic:: Exercise

    Write a program to search your PYTHONPATH for the module ``site.py``.

:ref:`path_site`
