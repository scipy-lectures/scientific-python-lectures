Scientific computing: what is it?
=================================

..
    .. image:: phd053104s.png
      :align: center

The scientist's needs
---------------------

* Acquire data (simulation, experiment control)

* Manipulate and process data.

* Visualize results... to understand what we are doing!

* Communicate on results: produce figures for reports or publications,
  write presentations.

Specifications
--------------

* Rich collection of already existing **bricks** corresponding to classical
  numerical methods or basic actions: we don't want to re-program the drawing of
  a curve, a Fourier transform or a fitting algorithm. Don't reinvent the wheel!

* Easy to learn: computer science neither is our job nor our education. We want to
  be able to draw a curve, smooth a signal, do a Fourier transform without
  spending hours finding out how to do.

* Easy communication with collaborators, students, customers, to make the code
  live within a labo or company: the code should be as readable as a book.
  Thus, the language should contain as few syntax symbols or unneeded routines
  that would divert the reader from the mathematical or scientific understanding
  of the code.

* Efficient code that executes quickly... But needless to say that a very fast
  code becomes useless if we spend too much time writing it. So, we need both a quick
  development time and a quick execution time.

* A single environment/language for eveything, if possible, to avoid learning
  new software for each new problem.

Existing solutions
------------------

Which solutions do the scientists use to work?

**Compiled languages: C, C++, Fortran, etc.**

* Advantages:

  * Very fast. Very optimized compilers. For heavy computations, it's difficult
    to surpass these languages.

  * Some very optimized scientific libraries have been written for these
    languages. Ex: blas (vector/matric operations)

* Drawbacks:

  * Painful usage: no interactivity during development,
    mandatory compilation steps, verbose syntax (&, ::, }}, ; etc.),
    manual memory management (delicate in C). These are **difficult
    languages** for non computer scientists.

**Scripting languages: Matlab**

* Advantages: 

  * Very rich collection of libraries with numerous algorithms, for many
    different domains. Fast execution because these libraries are often written
    in a compiled language.

  * Pleasant development environment: comprehensive and well organized help,
    integrated editor, etc.

  * commercial support availability

* Drawbacks: 

  * Base language is quite poor and can become restrictive for advanced users.

  * high price

**Other script languages: Scilab, Octave, Igor, R, IDL, etc.**

* Advantages:

  *  open-source, free, or at least cheaper than Matlab.

  * Some features can be very advanced (statistics in R, figures in Igor, etc.)

* Drawbacks:

  * fewer available algorithms than in Matlab, and the language
    is not more advanced.

  * Software dedicated to one domain. Ex: Gnuplot or xmgrace
    to draw curves. These programs are very powerful, but they are tied to 
    a single type of usage: plotting. It's a pity to be forced to learn a program
    just for that.

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

  * Scientific libraries don't provide a toolset as comprehensive as Matlab.
    But some of them can be more full-featured or better designed than Matlab.


