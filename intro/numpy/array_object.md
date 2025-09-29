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

# The NumPy array object

```{code-cell}
:tags: [hide-input]

# Our usual import.
import numpy as np
```

## What are NumPy and NumPy arrays?

### NumPy arrays

+++

**NumPy** provides:

- An extension package to Python for multi-dimensional arrays.
- An implementation that is closer to hardware (efficiency).
- Package designed for scientific computation (convenience).
- An implementation of _array oriented computing_.

```{code-cell}
import numpy as np

a = np.array([0, 1, 2, 3])
a
```

::: {note}
:class: dropdown

For example, An array containing:

- values of an experiment/simulation at discrete time steps

- signal recorded by a measurement device, e.g. sound wave

- pixels of an image, grey-level or colour

- 3-D data measured at different X-Y-Z positions, e.g. MRI scan

- ...
  :::

**Why it is useful:** Memory-efficient container that provides fast numerical
operations.

```{code-cell}
L = range(1000)
%timeit [i**2 for i in L]
```

```{code-cell}
a = np.arange(1000)
%timeit a**2
```

<!---
extension package to Python to support multidimensional arrays
-->
<!---
diagram, import conventions
-->
<!---
scope of this tutorial: drill in features of array manipulation in
Python, and try to give some indication on how to get things done
in good style
-->
<!---
a fixed number of elements (cf. certain exceptions)
-->
<!---
each element of same size and type
-->
<!---
efficiency vs. Python lists
-->

### NumPy Reference documentation

**On the web**:

<https://numpy.org/doc/>

**Interactive help:**

```ipython
In [5]: np.array?
String Form:<built-in function array>
Docstring:
array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0, ...
```

You can also use the Python builtin `help` command to show the docstring for a function:

```{code-cell}
help(np.array)
```

#### Looking for something:

```ipython
In [6]: np.con*?
np.concatenate
np.conj
np.conjugate
np.convolve
```

### Import conventions

The recommended convention to import NumPy is:

```{code-cell}
import numpy as np
```

## Creating arrays

### Manual construction of arrays

- **1-D**:

```{code-cell}
a = np.array([0, 1, 2, 3])
a
```

```{code-cell}
a.ndim
```

```{code-cell}
a.shape
```

```{code-cell}
len(a)
```

- **2-D, 3-D, ...**:

```{code-cell}
b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
b
```

```{code-cell}
b.ndim
```

```{code-cell}
b.shape
```

```{code-cell}
len(b)     # returns the size of the first dimension
```

```{code-cell}
c = np.array([[[1], [2]], [[3], [4]]])
c
```

```{code-cell}
c.shape
```

::: {exercise-start}
:label: np-ao-first-ex
:class: dropdown
:::

- Create a simple two dimensional array. First, redo the examples
  from above. And then create your own: how about odd numbers
  counting backwards on the first row, and even numbers on the second?
- Use the functions {func}`len`, {func}`numpy.shape` on these arrays.
  How do they relate to each other? And to the `ndim` attribute of
  the arrays?

::: {exercise-end}
:::

+++

### Functions for creating arrays

::: {note}
:class: dropdown

In practice, we rarely enter items one by one...
:::

**Evenly spaced**:

```{code-cell}
a = np.arange(10) # 0 .. n-1  (!)
a
```

```{code-cell}
b = np.arange(1, 9, 2) # start, end (exclusive), step
b
```

— or **by number of points**

```{code-cell}
c = np.linspace(0, 1, 6)   # start, end, num-points
c
```

```{code-cell}
d = np.linspace(0, 1, 5, endpoint=False)
d
```

**Common arrays**

```{code-cell}
a = np.ones((3, 3))  # reminder: (3, 3) is a tuple
a
```

```{code-cell}
b = np.zeros((2, 2))
b
```

```{code-cell}
c = np.eye(3)
c
```

```{code-cell}
d = np.diag(np.array([1, 2, 3, 4]))
d
```

- {mod}`numpy.random`: random numbers (Mersenne Twister PRNG):

```{code-cell}
rng = np.random.default_rng(27446968)
a = rng.random(4)       # uniform in [0, 1]
a
```

```{code-cell}
b = rng.standard_normal(4)      # Gaussian
b
```

::: {exercise-start}
:label: np-ao-func1-ex
:class: dropdown
:::

- Experiment with `arange`, `linspace`, `ones`, `zeros`, `eye` and
  `diag`.
- Create different kinds of arrays with random numbers.
- Try setting the seed before creating an array with random values.
- Look at the function `np.empty`. What does it do? When might this be
  useful?

::: {exercise-end}
:::

::: {exercise-start}
:label: np-ao-func2-ex
:class: dropdown
:::

- construct an array containing: 1 2 3 4 5
- construct an array containing: -5, -4, -3, -2, -1
- Construct: 2 4 6 8
- Construct 15 equispaced numbers in range [0, 10]

::: {exercise-end}
:::

::: {solution-start} np-ao-func2-ex
:class: dropdown
:::

```{code-cell}
np.arange(1, 6)
```

```{code-cell}
np.arange(-5, 0)
```

```{code-cell}
np.arange(2, 10, 2)
```

```{code-cell}
np.linspace(0, 10, 15)
```

::: {solution-end}
:::

+++

## Basic data types

You may have noticed that, in some instances, array elements are displayed with
a trailing dot (e.g. `2.` vs `2`). This is due to a difference in the
data-type used:

```{code-cell}
a = np.array([1, 2, 3])
a.dtype
```

```{code-cell}
b = np.array([1., 2., 3.])
b.dtype
```

::: {note}
:class: dropdown

Different data-types allow us to store data more compactly in memory,
but most of the time we simply work with floating point numbers.
Note that, in the example above, NumPy auto-detects the data-type
from the input.
:::

You can explicitly specify which data-type you want:

```{code-cell}
c = np.array([1, 2, 3], dtype=float)
c.dtype
```

The **default** data type is floating point:

```{code-cell}
a = np.ones((3, 3))
a.dtype
```

There are also other types:

+++

## Bool

```{code-cell}
e = np.array([True, False, False, True])
e.dtype
```

## Strings

```{code-cell}
f = np.array(['Bonjour', 'Hello', 'Hallo'])
f.dtype     # <--- strings containing max. 7 letters
```

## Much more:

- `int32`
- `int64`
- `uint32`
- `uint64`
- ...

<!---
XXX: mention: astype
-->

## Basic visualization

Now that we have our first data arrays, we are going to visualize them.

Start by launching IPython:

```bash
$ ipython # or ipython3 depending on your install
```

Or the notebook:

```bash
$ jupyter notebook
```

If you are using IPython enable interactive plots with:

```{code-cell}
%matplotlib
```

Interactive plots are enabled automatically in the Jupyter Notebook.

_Matplotlib_ is a 2D plotting package. We can import its functions as below:

```{code-cell}
import matplotlib.pyplot as plt  # the tidy way
```

And then use (note that you have to use `show` explicitly if you have not enabled interactive plots with `%matplotlib`):

```{code-cell}
# Example data
x = np.linspace(0, 2 * np.pi)
y = np.cos(x)

plt.plot(x, y)       # line plot
plt.show()           # <-- shows the plot (not needed with interactive plots)
```

Or, if you have enabled interactive plots with `%matplotlib`:

```{code-cell}
plt.plot(x, y)       # line plot
```

- **1D plotting**:

```{code-cell}
x = np.linspace(0, 3, 20)
y = np.linspace(0, 9, 20)
plt.plot(x, y)       # line plot
```

```{code-cell}
plt.plot(x, y, 'o')  # dot plot
```

- **2D arrays** (such as images):

```{code-cell}
rng = np.random.default_rng(27446968)
image = rng.random((30, 30))
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()
```

:::{admonition} See also

More in the: {ref}`matplotlib chapter <matplotlib>`
:::

::: {exercise-start}
:label: np-ao-viz-ex
:class: dropdown
:::

- Plot some simple arrays: a cosine as a function of time and a 2D
  matrix.
- Try using the `gray` colormap on the 2D matrix.

::: {exercise-end}
:::

+++

## Indexing and slicing

The items of an array can be accessed and assigned to the same way as
other Python sequences (e.g. lists):

```{code-cell}
a = np.arange(10)
a
```

```{code-cell}
a[0], a[2], a[-1]
```

:::{warning}
Indices begin at 0, like other Python sequences (and C/C++).
In contrast, in Fortran or Matlab, indices begin at 1.
:::

The usual python idiom for reversing a sequence is supported:

```{code-cell}
a[::-1]
```

For multidimensional arrays, indices are tuples of integers:

```{code-cell}
a = np.diag(np.arange(3))
a
```

```{code-cell}
a[1, 1]
```

```{code-cell}
a[2, 1] = 10 # third line, second column
a
```

```{code-cell}
a[1]
```

:::{note}

- In 2D, the first dimension corresponds to **rows**, the second
  to **columns**.
- for multidimensional `a`, `a[0]` is interpreted by
  taking all elements in the unspecified dimensions.
  :::

**Slicing**: Arrays, like other Python sequences can also be sliced:

```{code-cell}
a = np.arange(10)
a
```

```{code-cell}
a[2:9:3] # [start:end:step]
```

Note that the last index is not included! :

```{code-cell}
a[:4]
```

All three slice components are not required: by default, `start` is 0,
`end` is the last and `step` is 1:

```{code-cell}
a[1:3]
```

```{code-cell}
a[::2]
```

```{code-cell}
a[3:]
```

A small illustrated summary of NumPy indexing and slicing...

![](../../pyximages/numpy_indexing.png)

You can also combine assignment and slicing:

```{code-cell}
a = np.arange(10)
a[5:] = 10
a
```

```{code-cell}
b = np.arange(5)
a[5:] = b[::-1]
a
```

::: {exercise-start}
:label: np-ao-slicing-ex
:class: dropdown
:::

- Try the different flavours of slicing, using `start`, `end` and
  `step`: starting from a linspace, try to obtain odd numbers
  counting backwards, and even numbers counting forwards.
- Reproduce the slices in the diagram above. You may
  use the following expression to create the array:

```python
np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
```

::: {exercise-end}
:::

+++

::: {exercise-start}
:label: np-ao-creation-ex
:class: dropdown
:::

An exercise on array creation.

Create the following arrays (with correct data types):

```python
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 2],
 [1, 6, 1, 1]]

[[0., 0., 0., 0., 0.],
 [2., 0., 0., 0., 0.],
 [0., 3., 0., 0., 0.],
 [0., 0., 4., 0., 0.],
 [0., 0., 0., 5., 0.],
 [0., 0., 0., 0., 6.]]
```

Par on course: 3 statements for each.

_Hint_: Individual array elements can be accessed similarly to a list,
e.g. `a[1]` or `a[1, 2]`.

_Hint_: Examine the docstring for `diag`.

::: {exercise-end}
:::

::: {solution-start} np-ao-creation-ex
:class: dropdown
:::

```{code-cell}
a = np.ones((4, 4), dtype=int)
a[3, 1] = 6
a[2, 3] = 2
a
```

```{code-cell}
b = np.zeros((6, 5))
b[1:] = np.diag(np.arange(2, 7))
b
```

::: {solution-end}
:::

::: {exercise-start}
:label: np-ao-tiling-ex
:class: dropdown
:::

Exercise on tiling for array creation.

Skim through the documentation for `np.tile`, and use this function
to construct the array:

```python
[[4, 3, 4, 3, 4, 3],
 [2, 1, 2, 1, 2, 1],
 [4, 3, 4, 3, 4, 3],
 [2, 1, 2, 1, 2, 1]]
```

::: {exercise-end}
:::

::: {solution-start} np-ao-tiling-ex
:class: dropdown
:::

```{code-cell}
block = np.array([[4, 3], [2, 1]])
a = np.tile(block, (2, 3))
a
```

::: {solution-end}
:::

+++

## Copies and views

A slicing operation creates a **view** on the original array, which is
just a way of accessing array data. Thus the original array is not
copied in memory. You can use `np.may_share_memory()` to check if two arrays
share the same memory block. Note however, that this uses heuristics and may
give you false positives.

**When modifying the view, the original array is modified as well**:

```{code-cell}
a = np.arange(10)
a
```

```{code-cell}
b = a[::2]
b
```

```{code-cell}
np.may_share_memory(a, b)
```

```{code-cell}
b[0] = 12
b
```

```{code-cell}
a   # (!)
```

```{code-cell}
a = np.arange(10)
c = a[::2].copy()  # force a copy
c[0] = 12
a
```

```{code-cell}
np.may_share_memory(a, c)
```

This behavior can be surprising at first sight... but it allows to save both
memory and time.

<!---
EXE: [1, 2, 3, 4, 5] -> [1, 2, 3]
-->
<!---
EXE: [1, 2, 3, 4, 5] -> [4, 5]
-->
<!---
EXE: [1, 2, 3, 4, 5] -> [1, 3, 5]
-->
<!---
EXE: [1, 2, 3, 4, 5] -> [2, 4]
-->
<!---
EXE: create an array [1, 1, 1, 1, 0, 0, 0]
-->
<!---
EXE: create an array [0, 0, 0, 0, 1, 1, 1]
-->
<!---
EXE: create an array [0, 1, 0, 1, 0, 1, 0]
-->
<!---
EXE: create an array [1, 0, 1, 0, 1, 0, 1]
-->
<!---
EXE: create an array [1, 0, 2, 0, 3, 0, 4]
-->
<!---
CHA: archimedean sieve
-->

### Worked example: Prime number sieve

![](images/prime-sieve.png)

Compute prime numbers in 0--99, with a sieve

First — construct a shape (100,) boolean array `is_prime`, filled with True in
the beginning:

```{code-cell}
is_prime = np.ones((100,), dtype=bool)
```

Next, cross out 0 and 1 which are not primes:

```{code-cell}
is_prime[:2] = 0
```

For each integer `j` starting from 2, cross out its higher multiples:

```{code-cell}
N_max = int(np.sqrt(len(is_prime) - 1))
for j in range(2, N_max + 1):
    is_prime[2*j::j] = False
```

Skim through `help(np.nonzero)`, and print the prime numbers

- Follow-up:

  - Move the above code into a script file named `prime_sieve.py`
  - Run it to check it works
  - Use the optimization suggested in [the sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes):

  > 1. Skip `j` which are already known to not be primes
  > 2. The first number to cross out is $j^2$

+++

## Fancy indexing

::: {note}
:class: dropdown

NumPy arrays can be indexed with slices, but also with boolean or
integer arrays (**masks**). This method is called _fancy indexing_.
It creates **copies not views**.
:::

### Using boolean masks

```{code-cell}
rng = np.random.default_rng(27446968)
a = rng.integers(0, 21, 15)
a
```

```{code-cell}
(a % 3 == 0)
```

```{code-cell}
mask = (a % 3 == 0)
extract_from_a = a[mask] # or,  a[a%3==0]
extract_from_a           # extract a sub-array with the mask
```

Indexing with a mask can be very useful to assign a new value to a sub-array:

```{code-cell}
a[a % 3 == 0] = -1
a
```

### Indexing with an array of integers

```{code-cell}
a = np.arange(0, 100, 10)
a
```

Indexing can be done with an array of integers, where the same index is repeated
several time:

```{code-cell}
a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list
```

New values can be assigned with this kind of indexing:

```{code-cell}
a[[9, 7]] = -100
a
```

**Tip**

When a new array is created by indexing with an array of integers, the
new array has the same shape as the array of integers:

```{code-cell}
a = np.arange(10)
idx = np.array([[3, 4], [9, 7]])
idx.shape
```

```{code-cell}
a[idx]
```

---

The image below illustrates various fancy indexing applications

![](../../pyximages/numpy_fancy_indexing.png)

::: {exercise-start}
:label: np-ao-fancy-ex
:class: dropdown
:::

- Again, reproduce the fancy indexing shown in the diagram above.
- Use fancy indexing on the left and array creation on the right to assign
  values into an array, for instance by setting parts of the array in
  the diagram above to zero.

::: {exercise-end}
:::

We can even use fancy indexing and {ref}`broadcasting <broadcasting>` at
the same time:

```{code-cell}
a = np.arange(12).reshape(3,4)
a
```

```{code-cell}
i = np.array([[0, 1], [1, 2]])
a[i, 2]  # same as a[i, 2 * np.ones((2, 2), dtype=int)]
```
