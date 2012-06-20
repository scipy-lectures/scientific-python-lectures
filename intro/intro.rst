Scientific computing with tools and workflow
=============================================

:authors: Fernando Perez, Emmanuelle Gouillart, Gaël Varoquaux

..
    .. image:: phd053104s.png
      :align: center

Why Python?
------------

The scientist's needs
.......................

* Get data (simulation, experiment control)

* Manipulate and process data.

* Visualize results... to understand what we are doing!

* Communicate on results: produce figures for reports or publications,
  write presentations.

Specifications
................

* Rich collection of already existing **bricks** corresponding to classical
  numerical methods or basic actions: we don't want to re-program the
  plotting of a curve, a Fourier transform or a fitting algorithm. Don't 
  reinvent the wheel!

* Easy to learn: computer science neither is our job nor our education. We 
  want to be able to draw a curve, smooth a signal, do a Fourier transform 
  in a few minutes.

* Easy communication with collaborators, students, customers, to make the code
  live within a labo or a company: the code should be as readable as a book.
  Thus, the language should contain as few syntax symbols or unneeded routines
  that would divert the reader from the mathematical or scientific understanding
  of the code.

* Efficient code that executes quickly... But needless to say that a very fast
  code becomes useless if we spend too much time writing it. So, we need both a   quick development time and a quick execution time.

* A single environment/language for everything, if possible, to avoid learning
  a new software for each new problem.

Existing solutions
...................

Which solutions do the scientists use to work?

**Compiled languages: C, C++, Fortran, etc.**

* Advantages:

  * Very fast. Very optimized compilers. For heavy computations, it's difficult
    to outperform these languages.

  * Some very optimized scientific libraries have been written for these
    languages. Ex: blas (vector/matrix operations)

* Drawbacks:

  * Painful usage: no interactivity during development,
    mandatory compilation steps, verbose syntax (&, ::, }}, ; etc.),
    manual memory management (tricky in C). These are **difficult
    languages** for non computer scientists.

**Scripting languages: Matlab**

* Advantages: 

  * Very rich collection of libraries with numerous algorithms, for many
    different domains. Fast execution because these libraries are often written
    in a compiled language.

  * Pleasant development environment: comprehensive and well organized help,
    integrated editor, etc.

  * Commercial support is available.

* Drawbacks: 

  * Base language is quite poor and can become restrictive for advanced users.

  * Not free.

**Other script languages: Scilab, Octave, Igor, R, IDL, etc.**

* Advantages:

  * Open-source, free, or at least cheaper than Matlab.

  * Some features can be very advanced (statistics in R, figures in Igor, etc.)

* Drawbacks:

  * fewer available algorithms than in Matlab, and the language
    is not more advanced.

  * Some software are dedicated to one domain. Ex: Gnuplot or xmgrace
    to draw curves. These programs are very powerful, but they are
    restricted to a single type of usage, such as plotting. 

**What about Python?**

* Advantages:
  
  * Very rich scientific computing libraries (a bit less than Matlab,
    though)
    
  * Well-thought language, allowing to write very readable and well structured
    code: we "code what we think".

  * Many libraries for other tasks than scientific computing (web server
    management, serial port access, etc.)

  * Free and open-source software, widely spread, with a vibrant community.

* Drawbacks:  

  * less pleasant development environment than, for example, Matlab. (More
    geek-oriented).

  * Not all the algorithms that can be found in more specialized
    software or toolboxes.

Scientific Python building blocks
-----------------------------------


Unlike Matlab, scilab or R, Python does not come with a pre-bundled set
of modules for scientific computing. Below are the basic building blocks
that can be combined to optain a scientific-computing environment:

* **Python**, a generic and modern computing language

    * Python language: data types (``string``, ``int``), flow control,
      data collections (lists, dictionaries), patterns, etc.

    * Modules of the standard library.

    * A large number of specialized modules or applications written in
      Python: web protocols, web framework, etc. ... and scientific
      computing.

    * Development tools (automatic tests, documentation generation)

  .. image:: snapshot_ipython.png
        :align: right
        :scale: 40

* **IPython**, an advanced **Python shell** http://ipython.scipy.org/moin/
 
* **Numpy** : provides powerful **numerical arrays** objects, and routines to
  manipulate them. http://www.numpy.org/

..  
    >>> import numpy as np
    >>> np.random.seed(4)

* **Scipy** : high-level data processing routines.
  Optimization, regression, interpolation, etc http://www.scipy.org/

  .. image:: random_c.jpg
        :scale: 40
        :align: right

* **Matplotlib** : 2-D visualization, "publication-ready" plots
  http://matplotlib.sourceforge.net/

  .. raw:: html

   <div style="clear: both"></div>

  .. image:: example_surface_from_irregular_data.jpg
        :scale: 60
        :align: right

* **Mayavi** : 3-D visualization
  http://code.enthought.com/projects/mayavi/
  

.. raw:: html

   <div style="clear: both"></div>

The interactive workflow: IPython and a text editor 
-----------------------------------------------------

**Interactive work to test and understand algorithm:** In this section, we
describe an interactive workflow with `IPython <http://ipython.org>`__ that is
handy to explore and understand algorithms.

Python is a general-purpose language. As such, there is not one blessed
environement to work into, and not only one way of using it. Although
this makes it harder for beginners to find their way, it makes it
possible for Python to be used to write programs, in web servers, or
embedded devices. 

.. note:: Reference document for this section:

    **IPython user manual:** http://ipython.org/ipython-doc/dev/index.html

Command line interaction
..........................

Start `ipython`:

.. sourcecode:: ipython

    In [1]: print('Hello world')
    Hello world

Getting help:

.. sourcecode:: ipython

    In [2]: print?
    Type:		builtin_function_or_method
    Base Class:	        <type 'builtin_function_or_method'>
    String Form:	<built-in function print>
    Namespace:	        Python builtin
    Docstring:
	print(value, ..., sep=' ', end='\n', file=sys.stdout)
	
	Prints the values to a stream, or to sys.stdout by default.
	Optional keyword arguments:
	file: a file-like object (stream); defaults to the current sys.stdout.
	sep:  string inserted between values, default a space.
	end:  string appended after the last value, default a newline.


Elaboration of the algorithm in an editor
..........................................

Create a file `my_file.py` in a text editor. Under EPD, you can use
`Scite`, available from the start menu. Under Python(x,y), you can use
Spyder. Under Ubuntu, if you don't already have your favorite editor, We
would advise installing `Stani's Python editor`. In the file, add the
following lines::

    s = 'Hello world'
    print(s) 

Now, you can run it in ipython and explore the resulting variables:

.. sourcecode:: ipython

    In [3]: %run my_file.py
    Hello word

    In [4]: s
    Out[4]: 'Hello word'

    In [5]: %whos
    Variable   Type    Data/Info
    ----------------------------
    s          str     Hello word


.. topic:: **From a script to functions**

    While it is tempting to work only with scripts, that is a file full 
    of instructions following each other, do plan to progressively evolve
    the script to a set of functions:

    * A script is not reusable, functions are.

    * Thinking in terms of functions helps breaking the problem in small 
      blocks.


.. :vim:spell:






