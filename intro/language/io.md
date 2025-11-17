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

# Input and Output

To be exhaustive, here are some information about input and output in
Python. Since we will use the NumPy methods to read and write files,
**you may skip this chapter at first reading**.

We write or read **strings** to/from files (other types must be converted to
strings). To write in a file:

```{code-cell}
f = open('workfile', 'w') # opens the workfile file
type(f)
```

```{code-cell}
f.write('This is a test \nand another test')
f.close()
```

To read from a file

```{code-cell}
f = open('workfile', 'r')
s = f.read()
print(s)
```

```{code-cell}
f.close()
```

:::{admonition} See also

For more details: <https://docs.python.org/3/tutorial/inputoutput.html>
:::

## Iterating over a file

```{code-cell}
f = open('workfile', 'r')

for line in f:
    print(line)
```

```{code-cell}
f.close()
```

### File modes

- Read-only: `r`

- Write-only: `w`

  - Note: Create a new file or _overwrite_ existing file.

- Append a file: `a`

- Read and Write: `r+`

- Binary mode: `b`

  - Note: Use for binary files, especially on Windows.
