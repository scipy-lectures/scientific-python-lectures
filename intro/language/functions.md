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

# Defining functions

## Function definition

```{code-cell}
def test():
    print('in test function')
```

```{code-cell}
test()
```

:::{warning}
Function blocks must be indented in the same way as other control-flow blocks.
:::

+++

## Return statement

Functions _always_ return values:

```{code-cell}
def disk_area(radius):
    return 3.14 * radius * radius
```

```{code-cell}
disk_area(1.5)
```

But - if you do not specify an explicit return value, functions return the
special Python value `None`.

```{code-cell}
def another_func(a):
    # Do nothing.
    # Notice there is no "return" statement.
    pass
```

```{code-cell}
result = another_func(10)
# Check whether result returned is None value.
result is None
```

:::{note}
Note the syntax to define a function:

- the `def` keyword;
- is followed by the function's **name**, then
- the arguments of the function are given between parentheses followed
  by a colon.
- the function body;
- and `return object` for optionally returning values.
  :::

+++

## Parameters

Mandatory parameters (positional arguments)

```{code-cell}
def double_it(x):
    return x * 2
```

```{code-cell}
double_it(3)
```

```{code-cell}
:tags: [raises-exception]

double_it()
```

Optional parameters (keyword or named arguments)

```{code-cell}
def double_it(x=2):
    return x * 2
```

```{code-cell}
double_it()
```

```{code-cell}
double_it(3)
```

Keyword arguments allow you to specify _default values_.

**Warning:** default values are evaluated when the function is defined, not
when it is called. This can be problematic when using mutable types (e.g.
dictionary or list) and modifying them in the function body, since the
modifications will be persistent across invocations of the function.

Using an immutable type in a keyword argument:

```{code-cell}
bigx = 10
def double_it(x=bigx):
    return x * 2
```

```{code-cell}
bigx = 1e9  # Now really big
double_it()
```

Using an mutable type in a keyword argument (and modifying it inside the
function body):

```{code-cell}
def add_to_dict(args={'a': 1, 'b': 2}):
    for i in args.keys():
        args[i] += 1
    print(args)
```

```{code-cell}
add_to_dict
```

```{code-cell}
add_to_dict()
```

```{code-cell}
add_to_dict()
```

```{code-cell}
add_to_dict()
```

More involved example implementing python's slicing:

```{code-cell}
def slicer(seq, start=None, stop=None, step=None):
    """Implement basic python slicing."""
    return seq[start:stop:step]
```

```{code-cell}
rhyme = 'one fish, two fish, red fish, blue fish'.split()
rhyme
```

```{code-cell}
slicer(rhyme)
```

```{code-cell}
slicer(rhyme, step=2)
```

```{code-cell}
slicer(rhyme, 1, step=2)
```

```{code-cell}
slicer(rhyme, start=1, stop=4, step=2)
```

The order of the keyword arguments does not matter:

```{code-cell}
slicer(rhyme, step=2, start=1, stop=4)
```

— but it is good practice to use the same ordering as the function's
definition.

_Keyword arguments_ are a very convenient feature for defining functions with
a variable number of arguments, especially when default values are to be used
in most calls to the function.

+++

## Passing by value

::: {note}
:class: dropdown

Can you modify the value of a variable inside a function? Most languages (C,
Java, ...) distinguish "passing by value" and "passing by reference". In
Python, such a distinction is somewhat artificial, and it is a bit subtle
whether your variables are going to be modified or not. Fortunately, there
exist clear rules.

Parameters to functions are references to objects, which are passed by
value. When you pass a variable to a function, python passes the
reference to the object to which the variable refers (the **value**).
Not the variable itself.
:::

If the **value** passed in a function is immutable, the function does not
modify the caller's variable. If the **value** is mutable, the function
may modify the caller's variable in-place:

```{code-cell}
def try_to_modify(x, y, z):
    x = 23
    y.append(42)
    z = [99] # new reference
    print(x)
    print(y)
    print(z)
```

```{code-cell}
a = 77    # immutable variable
b = [99]  # mutable variable
c = [28]
try_to_modify(a, b, c)
```

```{code-cell}
print(a)
```

```{code-cell}
print(b)
```

```{code-cell}
print(c)
```

Functions have a local variable table called a _local namespace_.

The variable `x` only exists within the function `try_to_modify`.

+++

## Global variables

Variables declared outside the function can be referenced within the function:

```{code-cell}
x = 5
def addx(y):
    return x + y
```

```{code-cell}
addx(10)
```

But these "global" variables cannot be modified within the function, unless
declared **global** in the function.

This doesn't work:

```{code-cell}
def setx(y):
    x = y
    print('x is %d' % x)
```

```{code-cell}
setx(10)
```

```{code-cell}
x
```

This works:

```{code-cell}
def setx(y):
    global x
    x = y
    print('x is %d' % x)
```

```{code-cell}
setx(10)
```

```{code-cell}
x
```

## Variable number of parameters

Special forms of parameters:

- `*args`: any number of positional arguments packed into a tuple
- `**kwargs`: any number of keyword arguments packed into a dictionary

```{code-cell}
def variable_args(*args, **kwargs):
    print('args is', args)
    print('kwargs is', kwargs)
```

```{code-cell}
variable_args('one', 'two', x=1, y=2, z=3)
```

## Docstrings

Documentation about what the function does and its parameters. General
convention:

```{code-cell}
def funcname(params):
    """Concise one-line sentence describing the function.

    Extended summary which can contain multiple paragraphs.
    """
    # function body
    pass
```

```{code-cell}
# Also assessible in Jupyter / IPython with "funcname?"
help(funcname)
```

:::{note}
**Docstring guidelines**

For the sake of standardization, the [Docstring
Conventions](https://peps.python.org/pep-0257) webpage documents the semantics
and conventions associated with Python docstrings.

Also, the NumPy and SciPy modules have defined a precise standard for
documenting scientific functions, that you may want to follow for your own
functions, with a `Parameters` section, an `Examples` section, etc. See
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>
:::

+++

## Functions are objects

Functions are first-class objects, which means they can be:

- assigned to a variable
- an item in a list (or any collection)
- passed as an argument to another function.

```{code-cell}
va = variable_args
va('three', x=1, y=2)
```

## Methods

Methods are functions attached to objects. You've seen these in our examples on
_lists_, _dictionaries_, _strings_, etc...

+++

## Exercises

::: {exercise-start}
:label: fibonacci-ex
:class: dropdown
:::

Write a function that displays the `n` first terms of the Fibonacci
sequence, defined by:

$$
\begin{align}
U_{0} &= 0 \\
U_{1} &= 1 \\
U_{n+2} &= U_{n+1} + U_{n}
\end{align}
$$

::: {exercise-end}
:::

::: {solution-start} fibonacci-ex
:class: dropdown
:::

```{code-cell}
def fib(n):
    """Display the n first terms of Fibonacci sequence"""
    a, b = 0, 1
    i = 0
    while i < n:
        print(b)
        a, b = b, a+b
        i +=1
```

```{code-cell}
fib(10)
```

::: {solution-end}
:::

::: {exercise-start}
:label: quicksort-ex
:class: dropdown
:::

Implement the [Quicksort algorithm, as defined by
Wikipedia](https://en.wikipedia.org/wiki/Quicksort)

```
function quicksort(array)
    var list less, greater
    if length(array) < 2
        return array
    select and remove a pivot value pivot from array
    for each x in array
        if x < pivot + 1 then append x to less
        else append x to greater
    return concatenate(quicksort(less), pivot, quicksort(greater))
```

::: {exercise-end}
:::

::: {solution-start} quicksort-ex
:class: dropdown
:::

```{code-cell}
def qsort(lst):
    """Quick sort: returns a sorted copy of the list."""
    if len(lst) <= 1:
        return lst
    pivot, rest = lst[0], lst[1:]

    # Could use list comprehension:
    # less_than      = [ lt for lt in rest if lt < pivot ]

    less_than = []
    for lt in rest:
        if lt < pivot:
            less_than.append(lt)

    # Could use list comprehension:
    # greater_equal  = [ ge for ge in rest if ge >= pivot ]

    greater_equal = []
    for ge in rest:
        if ge >= pivot:
            greater_equal.append(ge)
    return qsort(less_than) + [pivot] + qsort(greater_equal)
```

```{code-cell}
# And now check that qsort does sort:
assert qsort(range(10)) == list(range(10))
assert qsort(range(10)[::-1]) == list(range(10))
assert qsort([1, 4, 2, 5, 3]) == sorted([1, 4, 2, 5, 3])
```

::: {solution-end}
:::
