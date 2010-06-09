===========
 Solutions
===========


.. _pi_wallis:

The Pi Wallis Solution
----------------------

Compute the decimals of Pi using the Wallis formula:

.. literalinclude:: solutions/pi_wallis.py

.. _quick_sort:

The Quicksort Solution
----------------------

Implement the quicksort algorithm, as defined by wikipedia:

::

	function quicksort(array)
	    var list less, greater
	    if length(array) ≤ 1  
		return array  
	    select and remove a pivot value pivot from array
	    for each x in array
		if x ≤ pivot then append x to less
		else append x to greater
	    return concatenate(quicksort(less), pivot, quicksort(greater))

.. literalinclude:: solutions/quick_sort.py

.. _fibonacci:

Fibonacci sequence
------------------

Write a function that displays the ``n`` first terms of the Fibonacci
sequence, defined by:

* ``u_0 = 1; u_1 = 1``
* ``u_(n+2) = u_(n+1) + u_n``

::

    >>> def fib(n):    
    ...     """Display the n first terms of Fibonacci sequence"""
    ...     a, b = 0, 1
    ...     i = 0
    ...     while i < n:
    ...         print b
    ...         a, b = b, a+b
    ...         i +=1
    ...
    >>> fib(10)
    1
    1
    2
    3
    5
    8
    13
    21
    34
    55

.. _dir_sort:

The Directory Listing Solution
------------------------------

Implement a script that takes a directory name as argument, and
returns the list of '.py' files, sorted by name length.

**Hint:** try to understand the docstring of list.sort

.. literalinclude:: solutions/dir_sort.py

.. _data_file:

The Data File I/O Solution
--------------------------

Write a function that will load the column of numbers in ``data.txt``
and calculate the min, max and sum values.

Data file:

.. literalinclude:: solutions/data.txt

Solution:

.. literalinclude:: solutions/data_file.py

.. _path_site:

The PYTHONPATH Search Solution
------------------------------

Write a program to search your PYTHONPATH for the module ``site.py``.

.. literalinclude:: solutions/path_site.py

