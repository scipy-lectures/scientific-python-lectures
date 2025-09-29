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

# First steps

Start the **Ipython** shell (an enhanced interactive Python shell):

- by typing "ipython" from a Linux/Mac terminal, or from the Windows cmd shell,
- **or** by starting the program from a menu, e.g. the [Anaconda Navigator],
  the [Python(x,y)] menu if you have installed one of these
  scientific-Python suites.

::: {note}
:class: dropdown

If you don't have Ipython installed on your computer, other Python
shells are available, such as the plain Python shell started by
typing "python" in a terminal, or the Idle interpreter. However, we
advise to use the Ipython shell because of its enhanced features,
especially for interactive scientific computing.
:::

Once you have started the interpreter, type

```{code-cell}
print("Hello, world!")
```

::: {note}
:class: dropdown

The message "Hello, world!" is then displayed. You just executed your
first Python instruction, congratulations!
:::

To get yourself started, type the following stack of instructions

```{code-cell}
a = 3
b = 2*a
type(b)
```

```{code-cell}
print(b)
```

```{code-cell}
a*b
```

```{code-cell}
b = 'hello'
type(b)
```

```{code-cell}
b + b
```

```{code-cell}
2*b
```

::: {note}
:class: dropdown

Two variables `a` and `b` have been defined above. Note that one does
not declare the type of a variable before assigning its value. In C,
conversely, one should write:

```c
int a = 3;
```

In addition, the type of a variable may change, in the sense that at
one point in time it can be equal to a value of a certain type, and a
second point in time, it can be equal to a value of a different
type. `b` was first equal to an integer, but it became equal to a
string when it was assigned the value `'hello'`. Operations on
integers (`b=2*a`) are coded natively in Python, and so are some
operations on strings such as additions and multiplications, which
amount respectively to concatenation and repetition.
:::

[anaconda navigator]: https://anaconda.org/anaconda/anaconda-navigator
[python(x,y)]: https://python-xy.github.io/
