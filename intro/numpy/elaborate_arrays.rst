.. For doctests
   
   >>> import numpy as np
   >>> np.random.seed(0)
   >>> from matplotlib import pyplot as plt

.. currentmodule:: numpy

More elaborate arrays
======================

.. contents:: Section contents
    :local:
    :depth: 1


.. XXX: maybe some of this should go to the advanced chapter

More data types
---------------

Casting
........

"Bigger" type wins in mixed-type operations::

    >>> np.array([1, 2, 3]) + 1.5
    array([ 2.5,  3.5,  4.5])

Assignment never changes the type! ::

    >>> a = np.array([1, 2, 3])
    >>> a.dtype
    dtype('int64')
    >>> a[0] = 1.9     # <-- float is truncated to integer
    >>> a
    array([1, 2, 3])

Forced casts::

    >>> a = np.array([1.7, 1.2, 1.6])
    >>> b = a.astype(int)  # <-- truncates to integer
    >>> b
    array([1, 1, 1])

Rounding::

    >>> a = np.array([1.2, 1.5, 1.6, 2.5, 3.5, 4.5])
    >>> b = np.around(a)
    >>> b                    # still floating-point
    array([ 1., 2., 2., 2., 4., 4.])
    >>> c = np.around(a).astype(int)
    >>> c
    array([ 1, 2, 2, 2, 4, 4])

Different data type sizes
..........................

Integers (signed):

=================== ==============================================================
:class:`int8`        8 bits
:class:`int16`       16 bits
:class:`int32`       32 bits (same as :class:`int` on 32-bit platform)
:class:`int64`       64 bits (same as :class:`int` on 64-bit platform)
=================== ==============================================================

>>> np.array([1], dtype=int).dtype
dtype('int64')
>>> np.iinfo(np.int32).max, 2**31 - 1
(2147483647, 2147483647)
>>> np.iinfo(np.int64).max, 2**63 - 1
(9223372036854775807, 9223372036854775807L)

Unsigned integers:

=================== ==============================================================
:class:`uint8`       8 bits
:class:`uint16`      16 bits
:class:`uint32`      32 bits
:class:`uint64`      64 bits
=================== ==============================================================

>>> np.iinfo(np.uint32).max, 2**32 - 1
(4294967295, 4294967295)
>>> np.iinfo(np.uint64).max, 2**64 - 1
(18446744073709551615L, 18446744073709551615L)

Floating-point numbers:

=================== ==============================================================
:class:`float16`     16 bits
:class:`float32`     32 bits
:class:`float64`     64 bits (same as :class:`float`)
:class:`float96`     96 bits, platform-dependent (same as :class:`np.longdouble`)
:class:`float128`    128 bits, platform-dependent (same as :class:`np.longdouble`)
=================== ==============================================================

>>> np.finfo(np.float32).eps
1.1920929e-07
>>> np.finfo(np.float64).eps
2.2204460492503131e-16

>>> np.float32(1e-8) + np.float32(1) == 1
True
>>> np.float64(1e-8) + np.float64(1) == 1
False

Complex floating-point numbers:

=================== ==============================================================
:class:`complex64`   two 32-bit floats
:class:`complex128`  two 64-bit floats
:class:`complex192`  two 96-bit floats, platform-dependent
:class:`complex256`  two 128-bit floats, platform-dependent
=================== ==============================================================

.. topic:: Smaller data types

   If you don't know you need special data types, then you probably don't.

   Comparison on using ``float32`` instead of ``float64``:

   - Half the size in memory and on disk
   - Half the memory bandwidth required (may be a bit faster in some operations)

     .. sourcecode:: ipython

        In [1]: a = np.zeros((1e6,), dtype=np.float64)

        In [2]: b = np.zeros((1e6,), dtype=np.float32)

        In [3]: %timeit a*a
        1000 loops, best of 3: 1.78 ms per loop

        In [4]: %timeit b*b
        1000 loops, best of 3: 1.07 ms per loop

   - But: bigger rounding errors --- sometimes in surprising places
     (i.e., don't use them unless you really need them)


Structured data types
---------------------

====================================  ==
``sensor_code`` (4-character string)
``position`` (float)
``value`` (float)
====================================  ==

>>> samples = np.zeros((6,), dtype=[('sensor_code', 'S4'),
...                                 ('position', float), ('value', float)])
>>> samples.ndim
1
>>> samples.shape
(6,)
>>> samples.dtype.names
('sensor_code', 'position', 'value')

>>> samples[:] = [('ALFA',   1, 0.37), ('BETA', 1, 0.11), ('TAU', 1,   0.13),
...               ('ALFA', 1.5, 0.37), ('ALFA', 3, 0.11), ('TAU', 1.2, 0.13)]
>>> samples
array([('ALFA', 1.0, 0.37), ('BETA', 1.0, 0.11), ('TAU', 1.0, 0.13),
       ('ALFA', 1.5, 0.37), ('ALFA', 3.0, 0.11), ('TAU', 1.2, 0.13)], 
      dtype=[('sensor_code', '|S4'), ('position', '<f8'), ('value', '<f8')])

Field access works by indexing with field names::

    >>> samples['sensor_code']
    array(['ALFA', 'BETA', 'TAU', 'ALFA', 'ALFA', 'TAU'], 
          dtype='|S4')
    >>> samples['value']
    array([ 0.37,  0.11,  0.13,  0.37,  0.11,  0.13])
    >>> samples[0]
    ('ALFA', 1.0, 0.37)

    >>> samples[0]['sensor_code'] = 'TAU'
    >>> samples[0]
    ('TAU', 1.0, 0.37)

Multiple fields at once::

    >>> samples[['position', 'value']]
    array([(1.0, 0.37), (1.0, 0.11), (1.0, 0.13), (1.5, 0.37), (3.0, 0.11),
           (1.2, 0.13)], 
          dtype=[('position', '<f8'), ('value', '<f8')])

Fancy indexing works, as usual::

    >>> samples[samples['sensor_code'] == 'ALFA']
    array([('ALFA', 1.5, 0.37), ('ALFA', 3.0, 0.11)], 
          dtype=[('sensor_code', '|S4'), ('position', '<f8'), ('value', '<f8')])

.. note:: There are a bunch of other syntaxes for constructing structured
   arrays, see `here <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`__
   and `here <http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types>`__.


:class:`maskedarray`: dealing with (propagation of) missing data
------------------------------------------------------------------

* For floats one could use NaN's, but masks work for all types::

    >>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
    >>> x
    masked_array(data = [1 -- 3 --],
                 mask = [False  True False  True],
           fill_value = 999999)
    <BLANKLINE>

    >>> y = np.ma.array([1, 2, 3, 4], mask=[0, 1, 1, 1])
    >>> x + y
    masked_array(data = [2 -- -- --],
                 mask = [False  True  True  True],
           fill_value = 999999)
    <BLANKLINE>

* Masking versions of common functions::

    >>> np.ma.sqrt([1, -1, 2, -2])
    masked_array(data = [1.0 -- 1.41421356237 --],
                 mask = [False  True False  True],
           fill_value = 1e+20)
    <BLANKLINE>


.. note::

   There are other useful :ref:`array siblings <array_siblings>`


.. _memory_layout:

Under the hood: the memory layout of a numpy array
---------------------------------------------------

A numpy array is:

    block of memory + indexing scheme + data type descriptor

    - raw data
    - how to locate an element
    - how to interpret an element

.. image:: threefundamental.png

Block of memory
................

>>> x = np.array([1, 2, 3, 4], dtype=np.int32)
>>> x.data      # doctest: +ELLIPSIS
<read-write buffer for ..., size 16, offset 0 at ...>
>>> str(x.data)
'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00'

Memory address of the data::

    >>> x.__array_interface__['data'][0] # doctest: +SKIP
    159755776

Reminder: two :class:`ndarrays <ndarray>` may share the same memory::

    >>> x = np.array([1,2,3,4])
    >>> y = x[:]
    >>> x[0] = 9
    >>> y
    array([9, 2, 3, 4])
    >>> y.base is x
    True

Memory does not need to be owned by an :class:`ndarray`::

    >>> x = '\x01\x02\x03\x04'
    >>> y = np.frombuffer(x, dtype=np.int8)
    >>> y
    array([1, 2, 3, 4], dtype=int8)
    >>> y.data  # doctest: +ELLIPSIS
    <read-only buffer for ..., size 4, offset 0 at ...>
    >>> y.base is x
    True

    >>> y.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : True
      OWNDATA : False
      WRITEABLE : False
      ALIGNED : True
      UPDATEIFCOPY : False

The ``owndata`` and ``writeable`` flags indicate status of the memory
block.


Indexing scheme: strides
.........................

**The question**

  >>> x = np.array([[1, 2, 3],
  ...               [4, 5, 6],
  ...               [7, 8, 9]], dtype=np.int8)
  >>> str(x.data)
  '\x01\x02\x03\x04\x05\x06\x07\x08\t'

  At which byte in ``x.data`` does the item ``x[1, 2]`` begin?

**The answer** (in Numpy)

  - **strides**: the number of bytes to jump to find the next element
  - 1 stride per dimension

  >>> x.strides
  (3, 1)
  >>> byte_offset = 3*1 + 1*2   # to find x[1, 2]
  >>> x.data[byte_offset]
  '\x06'
  >>> x[1, 2]
  6

  - simple, **flexible**


.. rubric:: C and Fortran order

>>> x = np.array([[1, 2, 3],
...               [4, 5, 6],
...               [7, 8, 9]], dtype=np.int16, order='C')
>>> x.strides
(6, 2)
>>> str(x.data)
'\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00\x06\x00\x07\x00\x08\x00\t\x00'

* Need to jump 6 bytes to find the next row
* Need to jump 2 bytes to find the next column


>>> y = np.array(x, order='F')
>>> y.strides
(2, 6)
>>> str(y.data)
'\x01\x00\x04\x00\x07\x00\x02\x00\x05\x00\x08\x00\x03\x00\x06\x00\t\x00'

* Need to jump 2 bytes to find the next row
* Need to jump 6 bytes to find the next column


- Similarly to higher dimensions:

  - C: last dimensions vary fastest (= smaller strides)
  - F: first dimensions vary fastest

  .. math::

     \mathrm{shape} &= (d_1, d_2, ..., d_n)
     \\
     \mathrm{strides} &= (s_1, s_2, ..., s_n)
     \\
     s_j^C &= d_{j+1} d_{j+2} ... d_{n} \times \mathrm{itemsize}
     \\
     s_j^F &= d_{1} d_{2} ... d_{j-1} \times \mathrm{itemsize}

.. rubric:: Slicing

- *Everything* can be represented by changing only ``shape``, ``strides``,
  and possibly adjusting the ``data`` pointer!
- Never makes copies of the data

>>> x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
>>> y = x[::-1]
>>> y
array([6, 5, 4, 3, 2, 1], dtype=int32)
>>> y.strides
(-4,)

>>> y = x[2:]
>>> y.__array_interface__['data'][0] - x.__array_interface__['data'][0]
8

>>> x = np.zeros((10, 10, 10), dtype=np.float)
>>> x.strides
(800, 80, 8)
>>> x[::2,::3,::4].strides
(1600, 240, 32)

- Similarly, transposes never make copies (it just swaps strides)

>>> x = np.zeros((10, 10, 10), dtype=np.float)
>>> x.strides
(800, 80, 8)
>>> x.T.strides
(8, 80, 800)

.. rubric:: Reshaping

But: not all reshaping operations can be represented by playing with
strides.

>>> a = np.arange(6, dtype=np.int8).reshape(3, 2)
>>> b = a.T
>>> b.strides
(1, 2)

So far, so good. However:

>>> str(a.data)
'\x00\x01\x02\x03\x04\x05'
>>> b
array([[0, 2, 4],
       [1, 3, 5]], dtype=int8)
>>> c = b.reshape(3*2)
>>> c
array([0, 2, 4, 1, 3, 5], dtype=int8)

Here, there is no way to represent the array ``c`` given one stride
and the block of memory for ``a``. Therefore, the ``reshape``
operation needs to make a copy here.


Summary
........

- Numpy array: block of memory + indexing scheme + data type description

- Indexing: strides

  ``byte_position = np.sum(arr.strides * indices)``

- Various tricks can you do by playing with the strides (stuff for an
  advanced tutorial it is)

_____

While it is off topic in a chapter on numpy, let's take a moment to
recall good coding practice, which really do pay off in the long run:

.. topic:: Good practices

    * Explicit variable names (no need of a comment to explain what is in
      the variable)

    * Style: spaces after commas, around ``=``, etc.

      A certain number of rules for writing "beautiful" code (and, more
      importantly, using the same conventions as everybody else!) are
      given in the `Style Guide for Python Code
      <http://www.python.org/dev/peps/pep-0008>`_ and the `Docstring
      Conventions <http://www.python.org/dev/peps/pep-0257>`_ page (to
      manage help strings).

    * Except some rare cases, variable names and comments in English.


