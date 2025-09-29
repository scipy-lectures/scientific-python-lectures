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

# Standard Library

:::{note}
Reference document for this section:

- The Python Standard Library documentation:
  <https://docs.python.org/3/library/index.html>
- Python Essential Reference, David Beazley, Addison-Wesley Professional
  :::

## `os` module: operating system functionality

_"A portable way of using operating system dependent functionality."_

### Directory and file manipulation

Current directory:

```{code-cell}
import os
os.getcwd()
```

List a directory:

```{code-cell}
os.listdir(os.curdir)
```

Make a directory:

```{code-cell}
os.mkdir('junkdir')
'junkdir' in os.listdir(os.curdir)
```

Rename the directory:

```{code-cell}
os.rename('junkdir', 'foodir')
'junkdir' in os.listdir(os.curdir)
```

```{code-cell}
'foodir' in os.listdir(os.curdir)
```

```{code-cell}
os.rmdir('foodir')
'foodir' in os.listdir(os.curdir)
```

Delete a file:

```{code-cell}
fp = open('junk.txt', 'w')
fp.close()
'junk.txt' in os.listdir(os.curdir)
```

```{code-cell}
os.remove('junk.txt')
'junk.txt' in os.listdir(os.curdir)
```

### `os.path`: path manipulations

`os.path` provides common operations on pathnames.

```{code-cell}
fp = open('junk.txt', 'w')
fp.close()
a = os.path.abspath('junk.txt')
a
```

```{code-cell}
os.path.split(a)
```

```{code-cell}
os.path.dirname(a)
```

```{code-cell}
os.path.basename(a)
```

```{code-cell}
os.path.splitext(os.path.basename(a))
```

```{code-cell}
os.path.exists('junk.txt')
```

```{code-cell}
os.path.isfile('junk.txt')
```

```{code-cell}
os.path.isdir('junk.txt')
```

```{code-cell}
os.path.expanduser('~/local')
```

```{code-cell}
os.path.join(os.path.expanduser('~'), 'local', 'bin')
```

### Running an external command

```{code-cell}
return_code = os.system('ls')
```

:::{note}
Alternative to `os.system`

A noteworthy alternative to `os.system` is the [sh
module](https://amoffat.github.com/sh/). Which provides much more convenient
ways to obtain the output, error stream and exit code of the external command.

```python
import sh
com = sh.ls()

print(com)
basic_types.md    exceptions.md   oop.md               standard_library.md
control_flow.md   first_steps.md  python_language.md
demo2.py          functions.md    python-logo.png
demo.py           io.md           reusing_code.md

type(com)
Out[33]: str
```

:::

### Walking a directory

`os.path.walk` generates a list of filenames in a directory tree.

```{code-cell}
for dirpath, dirnames, filenames in os.walk(os.curdir):
    for fp in filenames:
        print(os.path.abspath(fp))
```

### Environment variables:

```ipython
In [2]: os.environ.keys()
Out[2]: KeysView(environ({'SHELL': '/bin/bash', 'PWD': '/home/mb312', 'LOGNAME': 'mb312', 'HOME': '/home/mb312', 'TERM': 'xterm', 'USER': 'mb312', 'SHLVL': '1', 'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin', 'MAIL': '/var/mail/mb312', '_': '/usr/bin/python3', 'LC_CTYPE': 'C.UTF-8'}))

In [3]: os.environ['SHELL']
Out[3]: '/bin/bash'
```

+++

## `shutil`: high-level file operations

The `shutil` provides useful file operations:

- `shutil.rmtree`: Recursively delete a directory tree.
- `shutil.move`: Recursively move a file or directory to another location.
- `shutil.copy`: Copy files or directories.

## `glob`: Pattern matching on files

The `glob` module provides convenient file pattern matching.

Find all files ending in `.txt`:

```{code-cell}
import glob
glob.glob('*.txt')
```

## `sys` module: system-specific information

System-specific information related to the Python interpreter.

**Which version of Python** are you running and where is it installed:

```{code-cell}
import sys
sys.platform
```

```{code-cell}
sys.version
```

```{code-cell}
sys.prefix
```

`sys.argv` gives you a **list of command line arguments** passed to a Python
script. It is useful when you call as script with e.g. `python my_script.py some arguments`. Inside the `my_arguments.py` script, you can get the passed arguments (here ['some', 'arguments']) with `sys.argv`.

`sys.path` is a list of strings that specifies the search path for
modules. Initialized from `PYTHONPATH`:

```{code-cell}
sys.path
```

## `pickle`: easy persistence

Useful to store arbitrary objects to a file. Not safe or fast!

```{code-cell}
import pickle
l = [1, None, 'Stan']
with open('test.pkl', 'wb') as file:
    pickle.dump(l, file)
```

```{code-cell}
with open('test.pkl', 'rb') as file:
    out = pickle.load(file)
```

```{code-cell}
out
```

## Exercises

::: {exercise-start}
:label: data-file-ex
:class: dropdown
:::

Write a function that will load the column of numbers in `data.txt` and
calculate the min, max and sum values. Use no modules except those in the
standard library; specifically, do not use Numpy.

{download}`data.txt`:

::: {literalinclude} data.txt

:::

::: {exercise-end}
:::

::: {solution-start} data-file-ex
:class: dropdown
:::

```{code-cell}
def load_data(filename):
    fp = open(filename)
    data_string = fp.read()
    fp.close()

    data = []
    for x in data_string.split():
        # Data is read in as a string. We need to convert it to floats
        data.append(float(x))

    # Could instead use the following one line with list comprehensions!
    # data = [float(x) for x in data_string.split()]
    return data
```

```{code-cell}
data = load_data("data.txt")
# Python provides these basic math functions.
print(f"min: {min(data):f}")
print(f"max: {max(data):f}")
print(f"sum: {sum(data):f}")
```

::: {solution-end}
:::

::: {exercise-start}
:label: dir-sort-ex
:class: dropdown
:::

Implement a _script_ that takes a directory name as argument, and
returns the list of '.py' files, sorted by name length.

**Hint:** try to understand the docstring of list.sort

::: {exercise-end}
:::

::: {solution-start} dir-sort-ex
:class: dropdown
:::

::: {literalinclude} solutions/dir_sort.py

:::

::: {solution-end}
:::

+++

::: {exercise-start}
:label: path-site-ex
:class: dropdown
:::

Write a program to search your `PYTHONPATH` for the module `site.py`.

::: {exercise-end}
:::

::: {solution-start} path-site-ex
:class: dropdown
:::

::: {literalinclude} solutions/path_site.py

:::

::: {solution-end}
:::
