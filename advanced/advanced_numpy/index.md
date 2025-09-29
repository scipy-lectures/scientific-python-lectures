---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.0-dev
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(advanced-numpy)=

# Advanced NumPy

**Author**: _Pauli Virtanen_

NumPy is at the base of Python's scientific stack of tools. Its purpose
to implement efficient operations on many items in a block of memory.
Understanding how it works in detail helps in making efficient use of its
flexibility, taking useful shortcuts.

This section covers:

- Anatomy of NumPy arrays, and its consequences. Tips and
  tricks.
- Universal functions: what, why, and what to do if you want
  a new one.
- Integration with other tools: NumPy offers several ways to
  wrap any data in an ndarray, without unnecessary copies.
- Recently added features, and what's in them: PEP
  3118 buffers, generalized ufuncs, ...

:::{admonition} Prerequisites

- NumPy
- Cython
- Pillow (Python imaging library, used in a couple of examples)
  :::

```{code-cell}
# Import Numpy module.
import numpy as np
# Import Matplotlib (for later).
import matplotlib.pyplot as plt
```

## Life of ndarray

### It's...

::: {admonition} What is an **ndarray**

An **ndarray** is:

- A block of memory and
- an indexing scheme and
- a data type descriptor.
  :::

Put another way, an ndarray has **raw data**, and algorithms to:

- locate an element
- interpret an element

::: {image} threefundamental.png
:::

```c
typedef struct PyArrayObject {
        PyObject_HEAD

        /* Block of memory */
        char *data;

        /* Data type descriptor */
        PyArray_Descr *descr;

        /* Indexing scheme */
        int nd;
        npy_intp *dimensions;
        npy_intp *strides;

        /* Other stuff */
        PyObject *base;
        int flags;
        PyObject *weakreflist;
} PyArrayObject;
```

### Block of memory

```{code-cell}
x = np.array([1, 2, 3], dtype=np.int32)
x.data
```

```{code-cell}
bytes(x.data)
```

Memory address of the data:

```{code-cell}
x.__array_interface__['data'][0]
```

The whole `__array_interface__`:

```{code-cell}
x.__array_interface__
```

Reminder: two {class}`ndarrays <ndarray>` may share the same memory:

```{code-cell}
x = np.array([1, 2, 3, 4])
y = x[:-1]
x[0] = 9
y
```

Memory does not need to be owned by an {class}`ndarray`:

```{code-cell}
x = b'1234'
```

x is a string (in Python 3 a bytes), we can represent its data as an
array of ints:

```{code-cell}
y = np.frombuffer(x, dtype=np.int8)
y.data
```

```{code-cell}
y.base is x
```

```{code-cell}
y.flags
```

The `owndata` and `writeable` flags indicate status of the memory
block.

:::{admonition} See also

[array interface](https://numpy.org/doc/stable/reference/arrays.interface.html)
:::

### Data types

#### The descriptor

{class}`dtype` describes a single item in the array:

::: {list-table} Dtypes

- - type
  - **scalar type** of the data, one of:
    - int8, int16, float64, _et al._ (fixed size)
    - str, unicode, void (flexible size)
- - itemsize
  - **size** of the data block
- - byteorder
  - **byte order**:
    - big-endian `>`
    - little-endian `<`
    - not applicable `|`
- - fields
  - sub-dtypes, if it's a **structured data type**
- - shape
  - shape of the array, if it's a **sub-array**

:::

```{code-cell}
np.dtype(int).type
```

```{code-cell}
np.dtype(int).itemsize
```

```{code-cell}
np.dtype(int).byteorder
```

#### Example: reading `.wav` files

The `.wav` file header:

|                 |                                       |
| --------------- | ------------------------------------- |
| chunk_id        | `"RIFF"`                              |
| chunk_size      | 4-byte unsigned little-endian integer |
| format          | `"WAVE"`                              |
| fmt_id          | `"fmt "`                              |
| fmt_size        | 4-byte unsigned little-endian integer |
| audio_fmt       | 2-byte unsigned little-endian integer |
| num_channels    | 2-byte unsigned little-endian integer |
| sample_rate     | 4-byte unsigned little-endian integer |
| byte_rate       | 4-byte unsigned little-endian integer |
| block_align     | 2-byte unsigned little-endian integer |
| bits_per_sample | 2-byte unsigned little-endian integer |
| data_id         | `"data"`                              |
| data_size       | 4-byte unsigned little-endian integer |

- 44-byte block of raw data (in the beginning of the file)
- ... followed by `data_size` bytes of actual sound data.

The `.wav` file header as a NumPy _structured_ data type:

```{code-cell}
wav_header_dtype = np.dtype([
    ("chunk_id", (bytes, 4)), # flexible-sized scalar type, item size 4
    ("chunk_size", "<u4"),    # little-endian unsigned 32-bit integer
    ("format", "S4"),         # 4-byte string
    ("fmt_id", "S4"),
    ("fmt_size", "<u4"),
    ("audio_fmt", "<u2"),     #
    ("num_channels", "<u2"),  # .. more of the same ...
    ("sample_rate", "<u4"),   #
    ("byte_rate", "<u4"),
    ("block_align", "<u2"),
    ("bits_per_sample", "<u2"),
    ("data_id", ("S1", (2, 2))), # sub-array, just for fun!
    ("data_size", "u4"),
    #
    # the sound data itself cannot be represented here:
    # it does not have a fixed size
   ])
```

:::{admonition} See also

wavreader.py
:::

```{code-cell}
wav_header_dtype['format']
```

```{code-cell}
wav_header_dtype.fields
```

```{code-cell}
wav_header_dtype.fields['format']
```

- The first element is the sub-dtype in the structured data, corresponding
  to the name `format`
- The second one is its offset (in bytes) from the beginning of the item

::: {exercise-start}
:label: sparse-dtype
:class: dropdown
:::

Make a "sparse" dtype by using offsets, and only some of the fields:

```{code-cell}
:tags: [raises-exception]

wav_header_dtype = np.dtype(dict(
  names=['format', 'sample_rate', 'data_id'],
  offsets=[offset_1, offset_2, offset_3], # counted from start of structure in bytes
  formats=list of dtypes for each of the fields,
))
```

and use that to read the sample rate, and `data_id` (as sub-array).

::: {exercise-end}
:::

```{code-cell}
f = open('data/test.wav', 'r')
wav_header = np.fromfile(f, dtype=wav_header_dtype, count=1)
f.close()
print(wav_header)
```

```{code-cell}
wav_header['sample_rate']
```

Let's try accessing the sub-array:

```{code-cell}
wav_header['data_id']
```

```{code-cell}
wav_header.shape
```

```{code-cell}
wav_header['data_id'].shape
```

When accessing sub-arrays, the dimensions get added to the end!

:::{note}

There are existing modules such as `wavfile`, `audiolab`,
etc. for loading sound data...

:::

#### Casting and re-interpretation/views

**casting**

- on assignment
- on array construction
- on arithmetic
- etc.
- and manually: `.astype(dtype)`

**data re-interpretation**

- manually: `.view(dtype)`

##### Casting

- Casting in arithmetic, in nutshell:

  - only type (not value!) of operands matters
  - largest "safe" type able to represent both is picked
  - scalars can "lose" to arrays in some situations

- Casting in general copies data:

```{code-cell}
x = np.array([1, 2, 3, 4], dtype=float)
x
```

```{code-cell}
y = x.astype(np.int8)
y
```

```{code-cell}
y + 1
```

```{code-cell}
:tags: [raises-exception]

y + 256
```

```{code-cell}
y + 256.0
```

```{code-cell}
y + np.array([256], dtype=np.int32)
```

- Casting on setitem: dtype of the array is not changed on item assignment:

```{code-cell}
y[:] = y + 1.5
y
```

:::{note}

Exact rules: see [NumPy documentation](https://numpy.org/doc/stable/reference/ufuncs.html#casting-rules)

:::

##### Re-interpretation / viewing

Let's say we have a data block in memory (4 bytes). For the moment (as indicated by the bars between the values), we'll consider this to be four `unit8` values:

|        |     |        |     |        |     |        |
| ------ | --- | ------ | --- | ------ | --- | ------ |
| `0x01` | │   | `0x02` | │   | `0x03` | │   | `0x04` |

However, we can interpret this block as:

- 4 of uint8 (as here), OR,
- 4 of int8, OR,
- 2 of int16, OR,
- 1 of int32, OR,
- 1 of float32, OR,
- ...

How to switch from one to another?

**Option 1: Switch the dtype**

```{code-cell}
x = np.array([1, 2, 3, 4], dtype=np.uint8)
x.dtype = "<i2"
x
```

|        |        |     |        |        |
| ------ | ------ | --- | ------ | ------ |
| `0x01` | `0x02` | │   | `0x03` | `0x04` |

:::{note}

little-endian: least significant byte is on the _left_ in memory

:::

**Option 2: Create a new view of type `uint32`, shorthand `i4`**

```{code-cell}
y = x.view("<i4")
y
```

|        |        |        |        |
| ------ | ------ | ------ | ------ |
| `0x01` | `0x02` | `0x03` | `0x04` |

**On Views**

- `.view()` makes _views_, does not copy (or alter) the memory block
- it only changes the dtype (and adjusts array shape):

```{code-cell}
x[1] = 5
y
```

```{code-cell}
y.base is x
```

**Mini-exercise: data re-interpretation**

:::{admonition} See also

view-colors.py
:::

::: {exercise-start}
:label: rgba-to-structured
:class: dropdown
:::

You have RGBA data in an array:

```{code-cell}
x = np.zeros((10, 10, 4), dtype=np.int8)
x[:, :, 0] = 1
x[:, :, 1] = 2
x[:, :, 2] = 3
x[:, :, 3] = 4
```

where the last three dimensions are the R, B, and G, and alpha channels.

How would you make a (10, 10) structured array with field names 'r', 'g', 'b',
'a' without copying data?

```{code-cell}
y = ...
```

```{code-cell}
:tags: [raises-exception]

assert (y['r'] == 1).all()
assert (y['g'] == 2).all()
assert (y['b'] == 3).all()
assert (y['a'] == 4).all()
```

::: {exercise-end}
:::

+++

::: {solution-start} rgba-to-structured
:class: dropdown
:::

```{code-cell}
y = x.view([('r', 'i1'),
            ('g', 'i1'),
            ('b', 'i1'),
            ('a', 'i1')]
             )[:, :, 0]
```

::: {solution-end}
:::

#### A puzzle

Another two arrays, each occupying exactly 4 bytes of memory:

```{code-cell}
x = np.array([[1, 3], [2, 4]], dtype=np.uint8)
x
```

```{code-cell}
y = x.transpose()
y
```

We view the elements of `x` (1 byte each) as `int16` (2 bytes each):

```{code-cell}
x.view(np.int16)
```

What is happening here? Take a look at the bytes stored in memory
by `x`:

```{code-cell}
x.tobytes()
```

The `\x` stands for heXadecimal, so what we are seeing is:

```
0x01 0x03 0x02 0x04
```

We ask NumPy to interpret these bytes as elements of dtype
`int16`—each of which occupies _two_ bytes in memory. Therefore,
`0x01 0x03` becomes the first `uint16` and `0x02 0x04` the
second.

You may then expect to see `0x0103` (259, when converting from
hexadecimal to decimal) as the first result. But your computer
likely stores most significant bytes first, and as such reads the
number as `0x0301` or 769 (go on and type `0x0301` into your Python
terminal to verify).

We can do the same on a copy of `y` (why doesn't it work on `y`
directly?):

```{code-cell}
y.copy().view(np.int16)
```

Can you explain these numbers, 513 and 1027, as well as the output
shape of the resulting array?

### Indexing scheme: strides

#### Main point

**The question**:

```{code-cell}
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.int8)
x.tobytes('A')
```

**The answer** (in NumPy)

- **strides**: the number of bytes to jump to find the next element
- 1 stride per dimension

```{code-cell}
x.strides
```

```{code-cell}
byte_offset = 3 * 1 + 1 * 2  # to find x[1, 2]
x.flat[byte_offset]
```

```{code-cell}
x[1, 2]
```

simple, **flexible**

##### C and Fortran order

:::{note}

The Python built-in {py:class}`bytes` returns bytes in C-order by default
which can cause confusion when trying to inspect memory layout. We use
{meth}`numpy.ndarray.tobytes` with `order=A` instead, which preserves
the C or F ordering of the bytes in memory.

:::

```{code-cell}
x = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int16, order='C')
x.strides
```

```{code-cell}
x.tobytes('A')
```

- Need to jump 6 bytes to find the next row
- Need to jump 2 bytes to find the next column

```{code-cell}
y = np.array(x, order='F')
y.strides
```

```{code-cell}
y.tobytes('A')
```

- Need to jump 2 bytes to find the next row
- Need to jump 4 bytes to find the next column

Similarly for higher dimensions:

- C: last dimensions vary fastest (= smaller strides)
- F: first dimensions vary fastest

$$
\begin{align}
\mathrm{shape} &= (d_1, d_2, ..., d_n)
\\
\mathrm{strides} &= (s_1, s_2, ..., s_n)
\\
s_j^C &= d_{j+1} d_{j+2} ... d_{n} \times \mathrm{itemsize}
\\
s_j^F &= d_{1} d_{2} ... d_{j-1} \times \mathrm{itemsize}
\end{align}
$$

**Now we can understand the behavior of `.view()`**

```{code-cell}
y = np.array([[1, 3], [2, 4]], dtype=np.uint8).transpose()
x = y.copy()
```

Transposition does not affect the memory layout of the data, only strides

```{code-cell}
x.strides
```

```{code-cell}
y.strides
```

```{code-cell}
x.tobytes('A')
```

```{code-cell}
y.tobytes('A')
```

- the results are different when interpreted as 2 of int16
- `.copy()` creates new arrays in the C order (by default)

+++

##### Slicing with integers

- _Everything_ can be represented by changing only `shape`, `strides`,
  and possibly adjusting the `data` pointer!
- Never makes copies of the data

```{code-cell}
x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
y = x[::-1]
y
```

```{code-cell}
y.strides
```

```{code-cell}
y = x[2:]
y.__array_interface__['data'][0] - x.__array_interface__['data'][0]
```

```{code-cell}
x = np.zeros((10, 10, 10), dtype=float)
x.strides
```

```{code-cell}
x[::2,::3,::4].strides
```

Similarly, transposes never make copies (it just swaps strides):

```{code-cell}
x = np.zeros((10, 10, 10), dtype=float)
x.strides
```

```{code-cell}
x.T.strides
```

But: not all reshaping operations can be represented by playing with
strides:

```{code-cell}
a = np.arange(6, dtype=np.int8).reshape(3, 2)
b = a.T
b.strides
```

So far, so good. However:

```{code-cell}
bytes(a.data)
```

```{code-cell}
b
```

```{code-cell}
c = b.reshape(3*2)
c
```

Here, there is no way to represent the array `c` given one stride
and the block of memory for `a`. Therefore, the `reshape`
operation needs to make a copy here.

(stride-manipulation-label)=

#### Example: fake dimensions with strides

**Stride manipulation**

```{code-cell}
from numpy.lib.stride_tricks import as_strided
help(as_strided)
```

:::{warning}
`as_strided` does **not** check that you stay inside the memory
block bounds...
:::

```{code-cell}
x = np.array([1, 2, 3, 4], dtype=np.int16)
as_strided(x, strides=(2*2, ), shape=(2, ))
```

```{code-cell}
x[::2]
```

:::{admonition} See also

stride-fakedims.py
:::

::: {exercise-start}
:label: harder-strides
:class: dropdown
:::

Convert this:

```{code-cell}
in_arr = np.array([1, 2, 3, 4], dtype=np.int8)
in_arr
```

to this:

```python
array([[1, 2, 3, 4],
       [1, 2, 3, 4],
       [1, 2, 3, 4]], dtype=np.int8)
```

using only `as_strided`.:

**Hint**: `byte_offset = stride[0]*index[0] + stride[1]*index[1] + ...`

::: {exercise-end}
:::

::: {admonition} Spoiler for strides exercise
:class: dropdown

Stride can also be _0_:

:::

+++

::: {solution-start} harder-strides
:class: dropdown
:::

```{code-cell}
x = np.array([1, 2, 3, 4], dtype=np.int8)
y = as_strided(x, strides=(0, 1), shape=(3, 4))
y
```

```{code-cell}
y.base.base is x
```

::: {solution-end}
:::

(broadcasting-advanced)=

#### Broadcasting

- Doing something useful with it: outer product
  of `[1, 2, 3, 4]` and `[5, 6, 7]`

```{code-cell}
x = np.array([1, 2, 3, 4], dtype=np.int16)
x2 = as_strided(x, strides=(0, 1*2), shape=(3, 4))
x2
```

```{code-cell}
y = np.array([5, 6, 7], dtype=np.int16)
y2 = as_strided(y, strides=(1*2, 0), shape=(3, 4))
y2
```

```{code-cell}
x2 * y2
```

**... seems somehow familiar ...**

```{code-cell}
x = np.array([1, 2, 3, 4], dtype=np.int16)
y = np.array([5, 6, 7], dtype=np.int16)
x[np.newaxis,:] * y[:,np.newaxis]
```

- Internally, array **broadcasting** is indeed implemented using 0-strides.

#### More tricks: diagonals

:::{admonition} See also

stride-diagonals.py
:::

::: {exercise-start}
:label: stride-diagonals
:class: dropdown
:::

Pick diagonal entries of the matrix: (assume C memory order):

```{code-cell}
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.int32)
```

Your task is to work out the correct strides for to get the diagonal of the array, as in:

```
x_diag = as_strided(x, shape=(3,), strides=(...,))
```

Next:

- Pick the first super-diagonal entries `[2, 6]`.
- And the sub-diagonals?

**Hint to the last two**: slicing first moves the point where striding starts
from.

::: {exercise-end}
:::

::: {solution-start} stride-diagonals
:class: dropdown
:::

Pick diagonals:

```{code-cell}
x_diag = as_strided(x, shape=(3, ), strides=((3+1)*x.itemsize,))
x_diag
```

Slice first, to adjust the data pointer:

```{code-cell}
as_strided(x[0, 1:], shape=(2, ), strides=((3+1)*x.itemsize, ))
```

```{code-cell}
as_strided(x[1:, 0], shape=(2, ), strides=((3+1)*x.itemsize, ))
```

::: {solution-end}
:::

#### Using np.diag

```{code-cell}
y = np.diag(x, k=1)
y
```

However,

```{code-cell}
y.flags.owndata
```

**Challenge**

::: {exercise-start}
:label: tensor-trace
:class: dropdown
:::

Compute the tensor trace:

```{code-cell}
x = np.arange(5*5*5*5).reshape(5, 5, 5, 5)
s = 0
for i in range(5):
    for j in range(5):
        s += x[j, i, j, i]
```

by striding, and using `sum()` on the result.

```{code-cell}
:tags: [raises-exception]

y = as_strided(x, shape=(5, 5), strides=(..., ...))
s2 = ...
assert s == s2
```

::: {exercise-end}
:::

::: {solution-start} tensor-trace
:class: dropdown
:::

```{code-cell}
y = as_strided(x, shape=(5, 5), strides=((5*5*5 + 5)*x.itemsize,
                                         (5*5 + 1)*x.itemsize))
s2 = y.sum()
s2
```

::: {solution-end}
:::

(cache-effects)=

#### CPU cache effects

Memory layout can affect performance:

```{code-cell}
x = np.zeros((20000,))
y = np.zeros((20000*67,))[::67]

x.shape, y.shape
```

```{code-cell}
%timeit np.median(x)
```

```{code-cell}
%timeit np.median(y)
```

```{code-cell}
x.strides, y.strides
```

::: {note}

** Are smaller strides faster**

::: {image} cpu-cacheline.png
:::

- CPU pulls data from main memory to its cache in blocks
- If many array items consecutively operated on fit in a single block (small stride):

  - $\Rightarrow$ fewer transfers needed
  - $\Rightarrow$ faster

:::

:::{admonition} See also

- [numexpr](https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/) is designed to mitigate
  cache effects when evaluating array expressions.
- [numba](https://numba.pydata.org/) is a compiler for Python code,
  that is aware of numpy arrays.
  :::

### Findings in dissection

::: {image} threefundamental.png
:::

- _memory block_: may be shared, `.base`, `.data`
- _data type descriptor_: structured data, sub-arrays, byte order,
  casting, viewing, `.astype()`, `.view()`
- _strided indexing_: strides, C/F-order, slicing w/ integers,
  `as_strided`, broadcasting, stride tricks, `diag`, CPU cache
  coherence

## Universal functions

### What are they?

- Ufunc performs an elementwise operation on all elements of an array.

  Examples: `np.add, np.subtract, scipy.special.*,` ...

- Automatically support: broadcasting, casting, ...
- The author of an ufunc only has to supply the elementwise operation,
  NumPy takes care of the rest.
- The elementwise operation needs to be implemented in C (or, e.g., Cython)

#### Parts of an Ufunc

**Part 1: provided by user**

```c
void ufunc_loop(void **args, int *dimensions, int *steps, void *data)
{
    /*
    * int8 output = elementwise_function(int8 input_1, int8 input_2)
    *
    * This function must compute the ufunc for many values at once,
    * in the way shown below.
    */
    char *input_1 = (char*)args[0];
    char *input_2 = (char*)args[1];
    char *output = (char*)args[2];
    int i;

    for (i = 0; i < dimensions[0]; ++i) {
        *output = elementwise_function(*input_1, *input_2);
        input_1 += steps[0];
        input_2 += steps[1];
        output += steps[2];
    }
}
```

**Part 2. The NumPy part, built by**

```c
char types[3]

types[0] = NPY_BYTE   /* type of first input arg */
types[1] = NPY_BYTE   /* type of second input arg */
types[2] = NPY_BYTE   /* type of third input arg */

PyObject *python_ufunc = PyUFunc_FromFuncAndData(
    ufunc_loop,
    NULL,
    types,
    1, /* ntypes */
    2, /* num_inputs */
    1, /* num_outputs */
    identity_element,
    name,
    docstring,
    unused)
```

A ufunc can also support multiple different input-output type combinations.

#### Making it easier

`ufunc_loop` is of very generic form, and NumPy provides pre-made ones

|                |                                                                          |
| -------------- | ------------------------------------------------------------------------ |
| `PyUfunc_f_f`  | `float elementwise_func(float input_1)`                                  |
| `PyUfunc_ff_f` | `float elementwise_func(float input_1, float input_2)`                   |
| `PyUfunc_d_d`  | `double elementwise_func(double input_1)`                                |
| `PyUfunc_dd_d` | `double elementwise_func(double input_1, double input_2)`                |
| `PyUfunc_D_D`  | `elementwise_func(npy_cdouble *input, npy_cdouble* output)`              |
| `PyUfunc_DD_D` | `elementwise_func(npy_cdouble *in1, npy_cdouble *in2, npy_cdouble* out)` |

- Only `elementwise_func` needs to be supplied
- ... except when your elementwise function is not in one of the above forms

### Exercise: building an ufunc from scratch

::: {exercise-start}
:label: mandelbrot-ufunc
:class: dropdown
:::

+++

The Mandelbrot fractal is defined by the iteration

$$
z \leftarrow z^2 + c
$$

where $c = x + i y$ is a complex number. This iteration is
repeated -- if $z$ stays finite no matter how long the iteration
runs, $c$ belongs to the Mandelbrot set.

First — make a ufunc called `mandel(z0, c)` that computes:

```python
z = z0
for k in range(iterations):
    z = z*z + c
```

Run for, say, 100 iterations or until `z.real**2 + z.imag**2 > 1000`.
Use it to determine which `c` are in the Mandelbrot set.

Our function is a simple one, so make use of the `PyUFunc_*` helpers.

Write it in Cython

:::{admonition} See also

mandel.pyx, mandelplot.py
:::

:::{only} latex

```{literalinclude} examples/mandel.pyx

```

:::

**Reminder**: some pre-made Ufunc loops:

|                |                                                                                   |
| -------------- | --------------------------------------------------------------------------------- |
| `PyUfunc_f_f`  | `float elementwise_func(float input_1)`                                           |
| `PyUfunc_ff_f` | `float elementwise_func(float input_1, float input_2)`                            |
| `PyUfunc_d_d`  | `double elementwise_func(double input_1)`                                         |
| `PyUfunc_dd_d` | `double elementwise_func(double input_1, double input_2)`                         |
| `PyUfunc_D_D`  | `elementwise_func(complex_double *input, complex_double* output)`                 |
| `PyUfunc_DD_D` | `elementwise_func(complex_double *in1, complex_double *in2, complex_double* out)` |

Type codes:

```
NPY_BOOL, NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT, NPY_INT, NPY_UINT,
NPY_LONG, NPY_ULONG, NPY_LONGLONG, NPY_ULONGLONG, NPY_FLOAT, NPY_DOUBLE,
NPY_LONGDOUBLE, NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE, NPY_DATETIME,
NPY_TIMEDELTA, NPY_OBJECT, NPY_STRING, NPY_UNICODE, NPY_VOID
```

::: {exercise-end}
:::

::: {solution-start} mandelbrot-ufunc
:class: dropdown
:::

```{literalinclude} examples/mandel-answer.pyx
:language: python
```

```{literalinclude} examples/mandelplot.py
:language: python
```

::: {image} mandelbrot.png
:::

:::{note}

Most of the boilerplate could be automated by these Cython modules:

<https://github.com/cython/cython/wiki/MarkLodato-CreatingUfuncs>

:::

**Several accepted input types**

E.g. supporting both single- and double-precision versions

```cython
cdef void mandel_single_point(double complex *z_in,
                              double complex *c_in,
                              double complex *z_out) nogil:
   ...

cdef void mandel_single_point_singleprec(float complex *z_in,
                                         float complex *c_in,
                                         float complex *z_out) nogil:
   ...

cdef PyUFuncGenericFunction loop_funcs[2]
cdef char input_output_types[3*2]
cdef void *elementwise_funcs[1*2]

loop_funcs[0] = PyUFunc_DD_D
input_output_types[0] = NPY_CDOUBLE
input_output_types[1] = NPY_CDOUBLE
input_output_types[2] = NPY_CDOUBLE
elementwise_funcs[0] = <void*>mandel_single_point

loop_funcs[1] = PyUFunc_FF_F
input_output_types[3] = NPY_CFLOAT
input_output_types[4] = NPY_CFLOAT
input_output_types[5] = NPY_CFLOAT
elementwise_funcs[1] = <void*>mandel_single_point_singleprec

mandel = PyUFunc_FromFuncAndData(
    loop_func,
    elementwise_funcs,
    input_output_types,
    2, # number of supported input types   <----------------
    2, # number of input args
    1, # number of output args
    0, # `identity` element, never mind this
    "mandel", # function name
    "mandel(z, c) -> computes iterated z*z + c", # docstring
    0 # unused
    )
```

::: {solution-end}
:::

### Generalized ufuncs

**ufunc**

`output = elementwise_function(input)`

Both `output` and `input` can be a single array element only.

**generalized ufunc**

`output` and `input` can be arrays with a fixed number of dimensions

For example, matrix trace (sum of diag elements):

```text
input shape = (n, n)
output shape = ()    #  i.e.  scalar

(n, n) -> ()
```

Matrix product:

```text
input_1 shape = (m, n)
input_2 shape = (n, p)
output shape  = (m, p)

(m, n), (n, p) -> (m, p)
```

- This is called the _"signature"_ of the generalized ufunc
- The dimensions on which the g-ufunc acts, are _"core dimensions"_

**Status in NumPy**

- g-ufuncs are in NumPy already ...
- new ones can be created with `PyUFunc_FromFuncAndDataAndSignature`
- most linear-algebra functions are implemented as g-ufuncs to enable working
  with stacked arrays:

```{code-cell}
import numpy as np
rng = np.random.default_rng(27446968)
np.linalg.det(rng.random((3, 5, 5)))
```

```{code-cell}
np.linalg._umath_linalg.det.signature
```

- matrix multiplication this way could be useful for operating on
  many small matrices at once
- Also see `tensordot` and `einsum`

<!---

The below gufunc examples were from `np.core.umath_tests`,
which is now deprecated. We need another source of example
gufuncs.  See the discussion at:

https://mail.python.org/archives/list/numpy-discussion@python.org/thread/ZG7AUSPYYUNSPQU3YUZS2XCFD7AT3BJP/

import numpy.core.umath_tests as ut
ut.matrix_multiply.signature

'(m,n),(n,p)->(m,p)'

x = np.ones((10, 2, 4))
y = np.ones((10, 4, 5))
ut.matrix_multiply(x, y).shape

(10, 2, 5)

* in both examples the last two dimensions became *core dimensions*,
and are modified as per the *signature*
* otherwise, the g-ufunc operates "elementwise"

-->

**Generalized ufunc loop**

Matrix multiplication `(m,n),(n,p) -> (m,p)`

```c
void gufunc_loop(void **args, int *dimensions, int *steps, void *data)
{
    char *input_1 = (char*)args[0];  /* these are as previously */
    char *input_2 = (char*)args[1];
    char *output = (char*)args[2];

    int input_1_stride_m = steps[3];  /* strides for the core dimensions */
    int input_1_stride_n = steps[4];  /* are added after the non-core */
    int input_2_strides_n = steps[5]; /* steps */
    int input_2_strides_p = steps[6];
    int output_strides_n = steps[7];
    int output_strides_p = steps[8];

    int m = dimension[1]; /* core dimensions are added after */
    int n = dimension[2]; /* the main dimension; order as in */
    int p = dimension[3]; /* signature */

    int i;

    for (i = 0; i < dimensions[0]; ++i) {
        matmul_for_strided_matrices(input_1, input_2, output,
                                    strides for each array...);

        input_1 += steps[0];
        input_2 += steps[1];
        output += steps[2];
    }
}
```

## Interoperability features

### Sharing multidimensional, typed data

Suppose you

1. Write a library than handles (multidimensional) binary data,
2. Want to make it easy to manipulate the data with NumPy, or whatever
   other library,
3. ... but would **not** like to have NumPy as a dependency.

Currently, 3 solutions:

1. the "old" buffer interface
2. the array interface
3. the "new" buffer interface ({pep}`3118`)

### The old buffer protocol

- Only 1-D buffers
- No data type information
- C-level interface; `PyBufferProcs tp_as_buffer` in the type object
- But it's integrated into Python (e.g. strings support it)

Mini-exercise using [Pillow](https://python-pillow.org/) (Python
Imaging Library):

:::{admonition} See also

pilbuffer.py
:::

::: {exercise-start}
:label: pil-buffer
:class: dropdown
:::

```{code-cell}
from PIL import Image
data = np.zeros((200, 200, 4), dtype=np.uint8)
data[:, :] = [255, 0, 0, 255] # Red
# In PIL, RGBA images consist of 32-bit integers whose bytes are [RR,GG,BB,AA]
data = data.view(np.int32).squeeze()
img = Image.frombuffer("RGBA", (200, 200), data, "raw", "RGBA", 0, 1)
img.save('test.png')
```

**The question**

What happens if `data` is now modified, and `img` saved again?

::: {exercise-end}
:::

### The old buffer protocol

Show how to exchange data between numpy and a library that only knows
the buffer interface:

```{code-cell}
# Make a sample image, RGBA format
x = np.zeros((200, 200, 4), dtype=np.uint8)
x[:, :, 0] = 255  # red
x[:, :, 3] = 255  # opaque

data_i32 = x.view(np.int32)  # Check that you understand why this is OK!

img = Image.frombuffer("RGBA", (200, 200), data_i32)
img.save("test_red.png")

# Modify the original data, and save again.
x[:, :, 1] = 255
img.save("test_recolored.png")
```

::: {image} test_red.png
:::

::: {image} test_recolored.png
:::

### Array interface protocol

- Multidimensional buffers
- Data type information present
- NumPy-specific approach; slowly deprecated (but not going away)
- Not integrated in Python otherwise

:::{admonition} See also

Documentation:
<https://numpy.org/doc/stable/reference/arrays.interface.html>
:::

```{code-cell}
x = np.array([[1, 2], [3, 4]])
x.__array_interface__
```

```{code-cell}
:tags: [hide-input]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
if not os.path.exists('data'): os.mkdir('data')
plt.imsave('data/test.png', data)
```

```{code-cell}
from PIL import Image
img = Image.open('data/test.png')
img.__array_interface__
```

```{code-cell}
x = np.asarray(img)
x.shape
```

:::{note}

A more C-friendly variant of the array interface is also defined.

:::

(array-siblings)=

## Array siblings: {class}`chararray`, {class}`MaskedArray`

### {class}`chararray <numpy.char.chararray>`: vectorized string operations

```{code-cell}
x = np.char.asarray(['a', '  bbb', '  ccc'])
x
```

```{code-cell}
x.upper()
```

### {class}`MaskedArray <numpy.ma.MaskedArray>` missing data

Masked arrays are arrays that may have missing or invalid entries.

For example, suppose we have an array where the fourth entry is invalid:

```{code-cell}
x = np.array([1, 2, 3, -99, 5])
```

One way to describe this is to create a masked array:

```{code-cell}
mx = np.ma.MaskedArray(x, mask=[0, 0, 0, 1, 0])
mx
```

Masked mean ignores masked data:

```{code-cell}
mx.mean()
```

```{code-cell}
np.mean(mx)
```

:::{warning}
Not all NumPy functions respect masks, for instance
`np.dot`, so check the return types.
:::

The `MaskedArray` returns a **view** to the original array:

```{code-cell}
mx[1] = 9
x
```

#### The mask

You can modify the mask by assigning:

```{code-cell}
mx[1] = np.ma.masked
mx
```

The mask is cleared on assignment:

```{code-cell}
mx[1] = 9
mx
```

The mask is also available directly:

```{code-cell}
mx.mask
```

The masked entries can be filled with a given value to get an usual
array back:

```{code-cell}
x2 = mx.filled(-1)
x2
```

The mask can also be cleared:

```{code-cell}
mx.mask = np.ma.nomask
mx
```

#### Domain-aware functions

The masked array package also contains domain-aware functions:

```{code-cell}
np.ma.log(np.array([1, 2, -1, -2, 3, -5]))
```

:::{note}

Streamlined and more seamless support for dealing with missing data
in arrays is making its way into NumPy 1.7. Stay tuned!

:::

**Example: Masked statistics**

Canadian rangers were distracted when counting hares and lynxes in
1903-1910 and 1917-1918, and got the numbers are wrong. (Carrot
farmers stayed alert, though.) Compute the mean populations over
time, ignoring the invalid numbers.

```{code-cell}
data = np.loadtxt('data/populations.txt')
populations = np.ma.MaskedArray(data[:,1:])
year = data[:, 0]
```

```{code-cell}
bad_years = (((year >= 1903) & (year <= 1910))
           | ((year >= 1917) & (year <= 1918)))
# '&' means 'and' and '|' means 'or'
populations[bad_years, 0] = np.ma.masked
populations[bad_years, 1] = np.ma.masked
```

```{code-cell}
populations.mean(axis=0)
```

```{code-cell}
populations.std(axis=0)
```

Note that Matplotlib knows about masked arrays:

```{code-cell}
plt.plot(year, populations, 'o-')
```

### `np.recarray`: purely convenience

```{code-cell}
arr = np.array([('a', 1), ('b', 2)], dtype=[('x', 'S1'), ('y', int)])
arr2 = arr.view(np.recarray)
arr2.x
```

```{code-cell}
arr2.y
```

## Summary

- Anatomy of the ndarray: data, dtype, strides.
- Universal functions: elementwise operations, how to make new ones
- Ndarray subclasses
- Various buffer interfaces for integration with other tools
- Recent additions: PEP 3118, generalized ufuncs

## Contributing to NumPy/SciPy

Get this tutorial: <https://www.euroscipy.org/talk/882>

### Why

- "There's a bug?"
- "I don't understand what this is supposed to do?"
- "I have this fancy code. Would you like to have it?"
- "I'd like to help! What can I do?"

### Reporting bugs

- Bug tracker (prefer **this**)

  - <https://github.com/numpy/numpy/issues>
  - <https://github.com/scipy/scipy/issues>
  - Click the "Sign up" link to get an account

- Mailing lists (<https://numpy.org/community/>)

  - If you're unsure
  - No replies in a week or so? Just file a bug ticket.

#### Good bug report

```text
Title: numpy.random.permutations fails for non-integer arguments

I'm trying to generate random permutations, using numpy.random.permutations

When calling numpy.random.permutation with non-integer arguments
it fails with a cryptic error message::

    >>> rng.permutation(12)
    array([ 2,  6,  4,  1,  8, 11, 10,  5,  9,  3,  7,  0])
    >>> rng.permutation(12.)
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "_generator.pyx", line 4844, in numpy.random._generator.Generator.permutation
    numpy.exceptions.AxisError: axis 0 is out of bounds for array of dimension 0

This also happens with long arguments, and so
np.random.permutation(X.shape[0]) where X is an array fails on 64
bit windows (where shape is a tuple of longs).

It would be great if it could cast to integer or at least raise a
proper error for non-integer types.

I'm using NumPy 1.4.1, built from the official tarball, on Windows
64 with Visual studio 2008, on Python.org 64-bit Python.
```

0. What are you trying to do?
1. **Small code snippet reproducing the bug** (if possible)

   - What actually happens
   - What you'd expect

2. Platform (Windows / Linux / OSX, 32/64 bits, x86/PPC, ...)
3. Version of NumPy/SciPy

```{code-cell}
print(np.__version__)
```

**Check that the following is what you expect**

```{code-cell}
print(np.__file__)
```

In case you have old/broken NumPy installations lying around.

If unsure, try to remove existing NumPy installations, and reinstall...

### Contributing to documentation

1. Documentation editor

   - <https://numpy.org/doc/stable/>

   - Registration

     - Register an account

     - Subscribe to `scipy-dev` mailing list (subscribers-only)

     - Problem with mailing lists: you get mail

       - But: **you can turn mail delivery off**

       - "change your subscription options", at the bottom of

         <https://mail.python.org/mailman3/lists/scipy-dev.python.org/>

     - Send a mail @ `scipy-dev` mailing list; ask for activation:

       ```text
       To: scipy-dev@scipy.org

       Hi,

       I'd like to edit NumPy/SciPy docstrings. My account is XXXXX

       Cheers,
       N. N.
       ```

   - Check the style guide:

     - <https://numpy.org/doc/stable/>
     - Don't be intimidated; to fix a small thing, just fix it

   - Edit

2. Edit sources and send patches (as for bugs)

3. Complain on the mailing list

### Contributing features

The contribution of features is documented on <https://numpy.org/doc/stable/dev/>

### How to help, in general

- Bug fixes always welcome!

  - What irks you most
  - Browse the tracker

- Documentation work

  - API docs: improvements to docstrings

    - Know some SciPy module well?

  - _User guide_

    - <https://numpy.org/doc/stable/user/>

- Ask on communication channels:

  - `numpy-discussion` list
  - `scipy-dev` list
