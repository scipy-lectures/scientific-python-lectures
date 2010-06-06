Input and Output
----------------

To be exhaustive, here are some informations about input and output in Python.
Since we will use the Numpy methods to read and write files, you can skip this
chapter in first read.

We write or read **strings** to/from files (other types must be converted to
strings). To write in a file::
::

    >>> f = open('workfile', 'w') # ouvre le fichier workfile
    >>> type(f)
    <type 'file'>
    >>> f.write('Ceci est un test \nEncore un test')
    >>> f.close()

To read from a file::

    >>> f = open('workfile', 'r')
    >>> s = f.read()
    >>> print s
    Ceci est un test 
    Encore un test
    >>> f.close()

For more details: http://docs.python.org/tutorial/inputoutput.html



