Input and Output
================

To be exhaustive, here are some information about input and output in Python.
Since we will use the Numpy methods to read and write files, you may skip this
chapter at first reading.

We write or read **strings** to/from files (other types must be converted to
strings). To write in a file::


    >>> f = open('workfile', 'w') # opens the workfile file
    >>> type(f)
    <type 'file'>
    >>> f.write('This is a test \nand another test')
    >>> f.close()

To read from a file

.. sourcecode:: ipython

    In [1]: f = open('workfile', 'r')

    In [2]: s = f.read()

    In [3]: print(s)
    This is a test 
    and another test

    In [4]: f.close()


For more details: http://docs.python.org/tutorial/inputoutput.html

Iterating over a file
~~~~~~~~~~~~~~~~~~~~~

.. sourcecode:: ipython

    In [6]: f = open('workfile', 'r')

    In [7]: for line in f:
    ...:     print line
    ...:     
    ...:     
    This is a test 

    and another test

    In [8]: f.close()

File modes
----------

* Read-only: ``r``
* Write-only: ``w``

  * Note: Create a new file or *overwrite* existing file.

* Append a file: ``a``
* Read and Write: ``r+``
* Binary mode: ``b``

  * Note: Use for binary files, especially on Windows.

