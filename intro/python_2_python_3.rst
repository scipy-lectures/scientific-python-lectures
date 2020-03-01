.. _python_2_python_3:

..
    >>> import numpy as np

Python 2 and Python 3
=====================

**Author**: *Pierre de Buyl*

.. topic:: Python 2 / 3

    Two major versions of Python exist, Python 2 and Python 3. Python 3 is the only
    supported version since january 2020 but **the two versions coexisted for about a decade
    of transition from Python 2 to Python 3.** The transition has come to and end as most
    software libraries drop Python 2 support.


A very short summary
--------------------

- **Python 2 is not supported by the Python Software Foundation since January 1st 2020.**
  There will be no more security patches for Python 2.7. See `Sunsetting Python 2
  <https://www.python.org/doc/sunset-python-2/>`_ and the `Python 3 Q & A
  <http://python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html>`_.

- **The default choice for everyone should be Python 3.** Choosing Python 2 should remain
  motivated by specific circumstances such as the dependency on unported libraries, with the
  understanding of the lack of official and community support.

- Python 2 and Python 3 share most of their syntax, enabling many programmers to port their
  programs. It is even possible to make many codes Python 2/3 compatible, even though there
  are limitations. This strategy was important in making the transition but is no longer
  recommended.

- The division of integers, 1/2 for instance, returns 0 under Python 2 (integer division,
  preserving type) and 0.5 under Python 3 (real division, promoting the integer to a
  floating point value). **A line of code can thus execute with no visible warning in both
  Python 2 and Python 3 but result in different outcomes.**

- Most scientific libraries have moved to Python 3. NumPy and many scientific software
  libraries dropped Python 2 support or will do so soon, see the `Python 3 statement
  <https://python3statement.org/>`_.


The SciPy Lecture Notes dropped Python 2 support in 2020. The release 2020.1 is almost
entirely Python 2 compatible, so you may use it as a reference if necessary. Know that
installing suitable packages will probably be challenging.



Breaking changes between Python 2 and Python 3
----------------------------------------------

Python 3 differs from Python 2 in several ways. We list the most relevant ones for
scientific users below.


Print function
..............

The most visible change is that ``print`` is not a "statement" anymore but a
function.

Whereas in Python 2 you could write ::

    >>> print 'hello, world' # doctest: +SKIP
    hello, world

in Python 3 you must write

    >>> print('hello, world')
    hello, world

By making :func:`print` a function, one can pass arguments such a file identifier where the
output will be sent.


Division
........

In Python 2, the division of two integers with a single slash character results in
floor-based integer division::

    >>> 1/2 # doctest: +SKIP
    0

In Python 3, the default behavior is to use real-valued division::

    >>> 1/2
    0.5

Integer division is given by a double slash operator::

    >>> 1//2
    0



Some new features in Python 3
-----------------------------


Changing ``print`` to a function and changing the result of the division operator were only
two of the motivations for Python 3. An incomplete list of the changes follows (there are
many more).

- By default, strings are in unicode. Sequence of arbitrary bytes use the type
  ``bytes``. This change leads to heavy porting for applications dealing with text.

- Since Python 3.5 and NumPy 1.10, there is a matrix multiplication operator::

    >>> np.eye(2) @ np.array([3, 4])
    array([3., 4.])

- Since Python 3.6, there is a new string formatting method, the `"f-string"
  <https://docs.python.org/3/reference/lexical_analysis.html#f-strings>`_::

     >>> name = 'SciPy'
     >>> print(f"Hello, {name}!")
     Hello, SciPy!

- In Python 2, ``range(N)`` return a list. For large value of N (for a loop iterating many
  times), this implies the creation of a large list in memory even though it is not
  necessary. Python 2 provided the alternative ``xrange``, that you will find in many
  scientific programs.

  In Python 3, :func:`range` return a dedicated type and does not allocate the memory for the
  corresponding list.

    >>> type(range(8))
    <class 'range'>
    >>> range(8)
    range(0, 8)

  You can transform the output of ``range`` into a list if necessary::

    >>> list(range(8))
    [0, 1, 2, 3, 4, 5, 6, 7]

