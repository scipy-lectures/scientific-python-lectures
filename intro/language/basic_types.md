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

# Basic types

## Numerical types

::: {note}
:class: dropdown

Python supports the following numerical, scalar types:
:::

+++

**Floats:**

```{code-cell}
c = 2.1
type(c)
```

**Complex:**

```{code-cell}
a = 1.5 + 0.5j
a.real
```

```{code-cell}
a.imag
```

```{code-cell}
type(1. + 0j)
```

**Booleans:**

```{code-cell}
3 > 4
```

```{code-cell}
test = (3 > 4)
test
```

```{code-cell}
type(test)
```

::: {note}
:class: dropdown

A Python shell can therefore replace your pocket calculator, with the
basic arithmetic operations `+`, `-`, `*`, `/`, `%` (modulo)
natively implemented
:::

```{code-cell}
7 * 3.
```

```{code-cell}
2**10
```

```{code-cell}
8 % 3
```

Type conversion (casting):

```{code-cell}
float(1)
```

## Containers

::: {note}
:class: dropdown

Python provides many efficient types of containers, in which
collections of objects can be stored.
:::

### Lists

::: {note}
:class: dropdown

A list is an ordered collection of objects, that may have different
types. For example:
:::

```{code-cell}
colors = ['red', 'blue', 'green', 'black', 'white']
type(colors)
```

Indexing: accessing individual objects contained in the list:

```{code-cell}
colors[2]
```

Counting from the end with negative indices:

```{code-cell}
colors[-1]
```

```{code-cell}
colors[-2]
```

:::{warning}
**Indexing starts at 0** (as in C), not at 1 (as in Fortran or Matlab)!
:::

Slicing: obtaining sublists of regularly-spaced elements:

```{code-cell}
colors
```

```{code-cell}
colors[2:4]
```

:::{Warning}
Note that `colors[start:stop]` contains the elements with indices `i`
such as `start<= i < stop` (`i` ranging from `start` to
`stop-1`). Therefore, `colors[start:stop]` has `(stop - start)` elements.
:::

**Slicing syntax**: `colors[start:stop:stride]`

::: {note}
:class: dropdown

All slicing parameters are optional:

```{code-cell}
colors
```

```{code-cell}
colors[3:]
```

```{code-cell}
colors[:3]
```

```{code-cell}
colors[::2]
```

:::

Lists are _mutable_ objects and can be modified:

```{code-cell}
colors[0] = 'yellow'
colors
```

```{code-cell}
colors[2:4] = ['gray', 'purple']
colors
```

::::{Note}
The elements of a list may have different types:

```{code-cell}
colors = [3, -200, 'hello']
colors
```

```{code-cell}
colors[1], colors[2]
```

::: {note}
:class: dropdown

For collections of numerical data that all have the same type, it
is often **more efficient** to use the `array` type provided by
the `numpy` module. A NumPy array is a chunk of memory
containing fixed-sized items. With NumPy arrays, operations on
elements can be faster because elements are regularly spaced in
memory and more operations are performed through specialized C
functions instead of Python loops.
:::
::::

::: {note}
:class: dropdown

Python offers a large panel of functions to modify lists, or query
them. Here are a few examples; for more details, see
<https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>
:::

Add and remove elements:

```{code-cell}
colors = ['red', 'blue', 'green', 'black', 'white']
colors.append('pink')
colors
```

```{code-cell}
colors.pop() # removes and returns the last item
```

```{code-cell}
colors
```

```{code-cell}
colors.extend(['pink', 'purple']) # extend colors, in-place
colors
```

```{code-cell}
colors = colors[:-2]
colors
```

Reverse:

```{code-cell}
rcolors = colors[::-1]
rcolors
```

```{code-cell}
rcolors2 = list(colors) # new object that is a copy of colors in a different memory area
rcolors2
```

```{code-cell}
rcolors2.reverse() # in-place; reversing rcolors2 does not affect colors
rcolors2
```

Concatenate and repeat lists:

```{code-cell}
rcolors + colors
```

```{code-cell}
rcolors * 2
```

**Sort:**

```{code-cell}
sorted(rcolors) # new object
```

```{code-cell}
rcolors
```

```{code-cell}
rcolors.sort()  # in-place
rcolors
```

:::{admonition} Methods and Object-Oriented Programming
The notation `rcolors.method()` (e.g. `rcolors.append(3)` and `colors.pop()`) is our
first example of object-oriented programming (OOP). Being a `list`, the
object `rcolors` owns the _method_ `function` that is called using the notation
**.**. No further knowledge of OOP than understanding the notation **.** is
necessary for going through this tutorial.
:::

:::{admonition} Discovering methods:
Reminder: in Ipython: tab-completion (press tab)

```python


rcolors.<TAB>
                 append()  count()   insert()  reverse()
                 clear()   extend()  pop()     sort()
                 copy()    index()   remove()
```

:::

### Strings

Different string syntaxes (simple, double or triple quotes):

```{code-cell}
s = 'Hello, how are you?'
s = "Hi, what's up"
s = '''Hello,
       how are you'''         # tripling the quotes allows the
                              # string to span more than one line
s = """Hi,
what's up?"""
```

However, if you try to run this code:

```text
'Hi, what's up?'
```

— you will get a syntax error. (Try it.) (Why?)

This syntax error can be avoided by enclosing the string in double quotes
instead of single quotes. Alternatively, one can prepend a backslash to the
second single quote. Other uses of the backslash are, e.g., the newline
character `\n` and the tab character `\t`.

::: {note}
:class: dropdown

Strings are collections like lists. Hence they can be indexed and
sliced, using the same syntax and rules.
:::

Indexing:

```{code-cell}
a = "hello"
a[0]
```

```{code-cell}
a[1]
```

```{code-cell}
a[-1]
```

::: {note}
:class: dropdown

(Remember that negative indices correspond to counting from the right
end.)
:::

Slicing:

```{code-cell}
a = "hello, world!"
a[3:6] # 3rd to 6th (excluded) elements: elements 3, 4, 5
```

```{code-cell}
a[2:10:2] # Syntax: a[start:stop:step]
```

```{code-cell}
a[::3] # every three characters, from beginning to end
```

::: {note}
:class: dropdown

Accents and special characters can also be handled as in Python 3
strings consist of Unicode characters.
:::

A string is an **immutable object** and it is not possible to modify its
contents. One may however create new strings from the original one.

```{code-cell}
:tags: [raises-exception]

a = "hello, world!"
a[2] = 'z'
```

```{code-cell}
a.replace('l', 'z', 1)
```

```{code-cell}
a.replace('l', 'z')
```

::: {note}
:class: dropdown

Strings have many useful methods, such as `a.replace` as seen
above. Remember the `a.` object-oriented notation and use tab
completion or `help(str)` to search for new methods.
:::

:::{admonition} See also

Python offers advanced possibilities for manipulating strings,
looking for patterns or formatting. The interested reader is referred to
<https://docs.python.org/3/library/stdtypes.html#string-methods> and
<https://docs.python.org/3/library/string.html#format-string-syntax>
:::

String formatting:

```{code-cell}
'An integer: %i; a float: %f; another string: %s' % (1, 0.1, 'string') # with more values use tuple after %
```

```{code-cell}
i = 102
filename = 'processing_of_dataset_%d.txt' % i   # no need for tuples with just one value after %
filename
```

### Dictionaries

::: {note}
:class: dropdown

A dictionary is basically an efficient table that **maps keys to
values**.
:::

```{code-cell}
tel = {'emmanuelle': 5752, 'sebastian': 5578}
tel['francis'] = 5915
tel
```

```{code-cell}
tel['sebastian']
```

```{code-cell}
tel.keys()
```

```{code-cell}
tel.values()
```

```{code-cell}
'francis' in tel
```

::: {note}
:class: dropdown

It can be used to conveniently store and retrieve values
associated with a name (a string for a date, a name, etc.). See
<https://docs.python.org/3/tutorial/datastructures.html#dictionaries>
for more information.

A dictionary can have keys (resp. values) with different types:

```{code-cell}
d = {'a':1, 'b':2, 3:'hello'}
d
```

:::

### More container types

**Tuples**

Tuples are basically immutable lists. The elements of a tuple are written
between parentheses, or just separated by commas:

```{code-cell}
t = 12345, 54321, 'hello!'
t[0]
```

```{code-cell}
t
u = (0, 2)
```

**Sets:** unordered, unique items:

```{code-cell}
s = set(('a', 'b', 'c', 'a'))
s
```

```{code-cell}
s.difference(('a', 'b'))
```

## Assignment operator

::: {note}
:class: dropdown

[Python library reference](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements)
says:

> Assignment statements are used to (re)bind names to values and to
> modify attributes or items of mutable objects.

In short, it works as follows (simple assignment):

1. an expression on the right hand side is evaluated, the corresponding
   object is created/obtained
2. a **name** on the left hand side is assigned, or bound, to the
   r.h.s. object
   :::

Things to note:

- A single object can have several names bound to it:

```{code-cell}
a = [1, 2, 3]
b = a
a
```

```{code-cell}
b
```

```{code-cell}
a is b
```

```{code-cell}
b[1] = 'hi!'
a
```

- to change a list _in place_, use indexing/slices:

```{code-cell}
a = [1, 2, 3]
a
```

```{code-cell}
a = ['a', 'b', 'c'] # Creates another object.
a
```

```{code-cell}
id(a)
```

```{code-cell}
a[:] = [1, 2, 3] # Modifies object in place.
a
```

```{code-cell}
id(a)
```

- the key concept here is **mutable vs. immutable**

  - mutable objects can be changed in place
  - immutable objects cannot be modified once created

:::{admonition} See also

A very good and detailed explanation of the above issues can
be found in David M. Beazley's article [Types and Objects in Python](https://www.informit.com/articles/article.aspx?p=453682).
:::
