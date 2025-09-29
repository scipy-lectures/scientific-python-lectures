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

# Advanced operations

## Polynomials

NumPy also contains polynomials in different bases:

For example, $3x^2 + 2x - 1$:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell}
p = np.poly1d([3, 2, -1])
p(0)
```

```{code-cell}
p.roots
```

```{code-cell}
p.order
```

```{code-cell}
x = np.linspace(0, 1, 20)
rng = np.random.default_rng()
y = np.cos(x) + 0.3*rng.random(20)
p = np.poly1d(np.polyfit(x, y, 3))

t = np.linspace(0, 1, 200) # use a larger number of points for smoother plotting
plt.plot(x, y, 'o', t, p(t), '-');
```

See <https://numpy.org/doc/stable/reference/routines.polynomials.poly1d.html>
for more.

### More polynomials (with more bases)

NumPy also has a more sophisticated polynomial interface, which supports
e.g. the Chebyshev basis.

$3x^2 + 2x - 1$:

```{code-cell}
p = np.polynomial.Polynomial([-1, 2, 3]) # coefs in different order!
p(0)
```

```{code-cell}
p.roots()
```

```{code-cell}
p.degree()  # In general polynomials do not always expose 'order'
```

Example using polynomials in Chebyshev basis, for polynomials in
range `[-1, 1]`:

```{code-cell}
x = np.linspace(-1, 1, 2000)
rng = np.random.default_rng()
y = np.cos(x) + 0.3*rng.random(2000)
p = np.polynomial.Chebyshev.fit(x, y, 90)
```

```{code-cell}
plt.plot(x, y, 'r.')
plt.plot(x, p(x), 'k-', lw=3)
```

The Chebyshev polynomials have some advantages in interpolation.

+++

## Loading data files

### Text files

Example: {download}`populations.txt <data/populations.txt>`.

```{code-cell}
data = np.loadtxt('data/populations.txt')
data
```

```{code-cell}
np.savetxt('pop2.txt', data)
data2 = np.loadtxt('pop2.txt')
```

:::{note}
If you have a complicated text file, what you can try are:

- `np.genfromtxt`
- Using Python's I/O functions and e.g. regexps for parsing
  (Python is quite well suited for this)
  :::

### Reminder: Navigating the filesystem with Jupyter and IPython

Show current directory:

```{code-cell}
pwd
```

Change to `data` subdirectory:

```{code-cell}
cd data
```

Show filesystem listing for current directory:

```{code-cell}
ls
```

Change back to containing directory.

```{code-cell}
cd ..
```

### Images

Using Matplotlib:

```{code-cell}
img = plt.imread('data/elephant.png')
img.shape, img.dtype
```

```{code-cell}
# Plot and save the original figure
plt.imshow(img)
plt.savefig('plot.png')
```

```{code-cell}
# Plot and save the red channel of the image.
plt.imsave('red_elephant.png', img[:,:,0], cmap=plt.cm.gray)
```

This saved only one channel (of RGB):

```{code-cell}
plt.imshow(plt.imread('red_elephant.png'))
```

Other libraries:

```{code-cell}
import imageio.v3 as iio

# Lower resolution (every sixth pixel in each dimension).
iio.imwrite('tiny_elephant.png', (img[::6,::6] * 255).astype(np.uint8))
plt.imshow(plt.imread('tiny_elephant.png'), interpolation='nearest')
```

### NumPy's own format

NumPy has its own binary format, not portable but with efficient I/O:

```{code-cell}
data = np.ones((3, 3))
np.save('pop.npy', data)
data3 = np.load('pop.npy')
```

### Well-known (& more obscure) file formats

- HDF5: [h5py](https://www.h5py.org/), [PyTables](https://www.pytables.org)
- NetCDF: `scipy.io.netcdf_file`, [netcdf4-python](https://code.google.com/archive/p/netcdf4-python), ...
- Matlab: `scipy.io.loadmat`, `scipy.io.savemat`
- MatrixMarket: `scipy.io.mmread`, `scipy.io.mmwrite`
- IDL: `scipy.io.readsav`

... if somebody uses it, there's probably also a Python library for it.

::: {exercise-start}
:label: npa-load-proc-ex
:class: dropdown
:::

Write code that loads data from {download}`populations.txt
<data/populations.txt>`: and drops the last column and the first 5 rows. Save
the smaller dataset to `pop2.txt`.

::: {exercise-end}
:::

::: {solution-start} npa-load-proc-ex
:class: dropdown
:::

```{code-cell}
data = np.loadtxt("data/populations.txt")
reduced_data = data[5:, :-1]
np.savetxt("pop2.txt", reduced_data)
```

::: {solution-end}
:::

<!---
loadtxt, savez, load, fromfile, tofile
-->
<!---
real life: point to HDF5, NetCDF, etc.
-->
<!---
EXE: use loadtxt to load a data file
-->
<!---
EXE: use savez and load to save data in binary format
-->
<!---
EXE: use tofile and fromfile to put and get binary data bytes in/from a file
follow-up: .view()
-->
<!---
EXE: parsing text files -- Python can do this reasonably well natively!
throw in the mix some random text file to be parsed (eg. PPM)
-->
<!---
EXE: advanced: read the data in a PPM file
-->

:::{admonition} NumPy internals
If you are interested in the NumPy internals, there is a good discussion in
{ref}`advanced-numpy`.
:::
