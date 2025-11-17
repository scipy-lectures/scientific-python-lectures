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

```{code-cell}
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
```

# More elaborate arrays

## More data types

### Casting

"Bigger" type wins in mixed-type operations:

```{code-cell}
np.array([1, 2, 3]) + 1.5
```

Assignment never changes the type!

```{code-cell}
a = np.array([1, 2, 3])
a.dtype
```

```{code-cell}
a[0] = 1.9     # <-- float is truncated to integer
a
```

Forced casts:

```{code-cell}
a = np.array([1.7, 1.2, 1.6])
b = a.astype(int)  # <-- truncates to integer
b
```

Rounding:

```{code-cell}
a = np.array([1.2, 1.5, 1.6, 2.5, 3.5, 4.5])
b = np.around(a)
b                    # still floating-point
```

```{code-cell}
c = np.around(a).astype(int)
c
```

### Different data type sizes

Integers (signed):

| Class   | Bits                                       |
| ------- | ------------------------------------------ |
| `int8`  | 8 bits                                     |
| `int16` | 16b its                                    |
| `int32` | 32 bits (same as `int` on 32-bit platform) |
| `int64` | 64 bits (same as `int` on 64-bit platform) |

```{code-cell}
np.array([1], dtype=int).dtype
```

```{code-cell}
np.iinfo(np.int32).max, 2**31 - 1
```

Unsigned integers:

| Class    | Bits    |
| -------- | ------- |
| `uint8`  | 8 bits  |
| `uint16` | 16 bits |
| `uint32` | 32 bits |
| `uint64` | 64 bits |

```{code-cell}
np.iinfo(np.uint32).max, 2**32 - 1
```

Floating-point numbers:

| Data Type  | Size (bits)                                            |
| ---------- | ------------------------------------------------------ |
| `float16`  | 16 bits                                                |
| `float32`  | 32 bits                                                |
| `float64`  | 64 bits (same as `float`)                              |
| `float96`  | 96 bits, platform-dependent (same as `np.longdouble`)  |
| `float128` | 128 bits, platform-dependent (same as `np.longdouble`) |

```{code-cell}
np.finfo(np.float32).eps
```

```{code-cell}
np.finfo(np.float64).eps
```

```{code-cell}
np.float32(1e-8) + np.float32(1) == 1
```

```{code-cell}
np.float64(1e-8) + np.float64(1) == 1
```

Complex floating-point numbers:

| Data Type    | Size (bits)                            |
| ------------ | -------------------------------------- |
| `complex64`  | two 32-bit floats                      |
| `complex128` | two 64-bit floats                      |
| `complex192` | two 96-bit floats, platform-dependent  |
| `complex256` | two 128-bit floats, platform-dependent |

:::{admonition} Smaller data types
If you don't know you need special data types, then you probably don't.

Comparison on using `float32` instead of `float64`:

- Half the size in memory and on disk

- Half the memory bandwidth required (may be a bit faster in some operations)

  ```ipython
  In [1]: a = np.zeros((int(1e6),), dtype=np.float64)

  In [2]: b = np.zeros((int(1e6),), dtype=np.float32)

  In [3]: %timeit a*a
  1000 loops, best of 3: 1.78 ms per loop

  In [4]: %timeit b*b
  1000 loops, best of 3: 1.07 ms per loop
  ```

- But: bigger rounding errors — sometimes in surprising places
  (i.e., don't use them unless you really need them)
  :::

## Structured data types

| Data Type     | Description        |
| ------------- | ------------------ |
| `sensor_code` | 4-character string |
| `position`    | float              |
| `value`       | float              |

```{code-cell}
samples = np.zeros((6,), dtype=[('sensor_code', 'S4'),
                                ('position', float), ('value', float)])
samples.ndim
```

```{code-cell}
samples.shape
```

```{code-cell}
samples.dtype.names
```

```{code-cell}
samples[:] = [('ALFA',   1, 0.37), ('BETA', 1, 0.11), ('TAU', 1,   0.13),
              ('ALFA', 1.5, 0.37), ('ALFA', 3, 0.11), ('TAU', 1.2, 0.13)]
samples
```

Field access works by indexing with field names:

```{code-cell}
samples['sensor_code']
```

```{code-cell}
samples['value']
```

```{code-cell}
samples[0]
```

```{code-cell}
samples[0]['sensor_code'] = 'TAU'
samples[0]
```

Multiple fields at once:

```{code-cell}
samples[['position', 'value']]
```

Fancy indexing works, as usual:

```{code-cell}
samples[samples['sensor_code'] == b'ALFA']
```

:::{note}
There are a bunch of other syntaxes for constructing structured
arrays, see [here](https://numpy.org/doc/stable/user/basics.rec.html)
and [here](https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types).
:::

## {class}`maskedarray`: dealing with (propagation of) missing data

- For floats one could use NaN's, but masks work for all types:

```{code-cell}
x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
x
```

```{code-cell}
y = np.ma.array([1, 2, 3, 4], mask=[0, 1, 1, 1])
x + y
```

- Masking versions of common functions:

```{code-cell}
np.ma.sqrt([1, -1, 2, -2])
```

:::{note}
There are other useful {ref}`array siblings <array-siblings>`
:::

---

While it is off topic in a chapter on NumPy, let's take a moment to
recall good coding practice, which really do pay off in the long run:

:::{admonition} Good practices

- Explicit variable names (no need of a comment to explain what is in
  the variable)

- Style: spaces after commas, around `=`, etc.

  A certain number of rules for writing "beautiful" code (and, more
  importantly, using the same conventions as everybody else!) are
  given in the [Style Guide for Python Code](https://peps.python.org/pep-0008) and the [Docstring
  Conventions](https://peps.python.org/pep-0257) page (to
  manage help strings).

- Except some rare cases, variable names and comments in English.
  :::
