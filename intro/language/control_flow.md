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

# Control Flow

Controls the order in which the code is executed.

## if/elif/else

```{code-cell}
if 2**2 == 4:
    print("Obvious!")
```

**Blocks are delimited by indentation**

::: {note}
:class: dropdown

Type the following lines in your Python interpreter, and be careful
to **respect the indentation depth**. The Jupyter / IPython shell automatically
increases the indentation depth after a colon `:` sign; to decrease the
indentation depth, go four spaces to the left with the Backspace key. Press the
Enter key twice to leave the logical block.
:::

```{code-cell}
a = 10
```

```{code-cell}
if a == 1:
    print(1)
elif a == 2:
    print(2)
else:
    print("A lot")
```

Indentation is compulsory in scripts as well. As an exercise, re-type the
previous lines with the same indentation in a script `condition.py`, and
execute the script with `run condition.py` in IPython.

+++

## for/range

Iterating with an index:

```{code-cell}
for i in range(4):
    print(i)
```

But most often, it is more readable to iterate over values:

```{code-cell}
for word in ('cool', 'powerful', 'readable'):
    print('Python is %s' % word)
```

## while/break/continue

Typical C-style while loop (Mandelbrot problem):

```{code-cell}
z = 1 + 1j
while abs(z) < 100:
    z = z**2 + 1
z
```

**More advanced features**

`break` out of enclosing for/while loop:

```{code-cell}
z = 1 + 1j
```

```{code-cell}
while abs(z) < 100:
    if z.imag == 0:
        break
    z = z**2 + 1
```

`continue` the next iteration of a loop.:

```{code-cell}
a = [1, 0, 2, 4]
for element in a:
    if element == 0:
        continue
    print(1. / element)
```

## Conditional Expressions

+++

### `if <OBJECT>:`

Evaluates to `False` for:

- any number equal to zero (0, 0.0, 0+0j)
- an empty container (list, tuple, set, dictionary, …)
- `False`, `None`

Evaluates to `True` for:

- everything else

Examples:

```{code-cell}
a = 10
if a:
    print("Evaluated to `True`")
else:
    print('Evaluated to `False')
```

```{code-cell}
a = []
if a:
    print("Evaluated to `True`")
else:
    print('Evaluated to `False')
```

### `a == b:`

Tests equality, with logics::

```{code-cell}
1 == 1.
```

### `a is b`

Tests identity: both sides **are the same object**:

```{code-cell}
a = 1
b = 1.
a == b
```

```{code-cell}
a is b
```

```{code-cell}
a = 'A string'
b = a
a is b
```

### `a in b`

For any collection `b`: `b` contains `a` :

```{code-cell}
b = [1, 2, 3]
2 in b
```

```{code-cell}
5 in b
```

If `b` is a dictionary, this tests that `a` is a key of `b`.

```{code-cell}
b = {'first': 0, 'second': 1}
# Tests for key.
'first' in b
```

```{code-cell}
# Does not test for value.
0 in b
```

## Advanced iteration

**Iterate over any sequence**:

You can iterate over any sequence (string, list, keys in a dictionary, lines in
a file, ...):

```{code-cell}
vowels = 'aeiouy'
```

```{code-cell}
for i in 'powerful':
    if i in vowels:
        print(i)
```

```{code-cell}
message = "Hello how are you?"
message.split() # returns a list
```

```{code-cell}
for word in message.split():
    print(word)
```

::: {note}
:class: dropdown

Few languages (in particular, languages for scientific computing) allow to
loop over anything but integers/indices. With Python it is possible to
loop exactly over the objects of interest without bothering with indices
you often don't care about. This feature can often be used to make
code more readable.
:::

:::{warning}
It is not safe to modify the sequence you are iterating over.
:::

### Keeping track of enumeration number

Common task is to iterate over a sequence while keeping track of the
item number.

We could use while loop with a counter as above. Or a for loop:

```{code-cell}
words = ('cool', 'powerful', 'readable')
for i in range(0, len(words)):
    print((i, words[i]))
```

But, Python provides a built-in function - `enumerate` - for this:

```{code-cell}
for index, item in enumerate(words):
    print((index, item))
```

### Looping over a dictionary

Use **items**:

```{code-cell}
d = {'a': 1, 'b':1.2, 'c':1j}
```

```{code-cell}
for key, val in d.items():
    print('Key: %s has value: %s' % (key, val))
```

## List Comprehensions

Instead of creating a list by means of a loop, one can make use
of a list comprehension with a rather self-explaining syntax.

```{code-cell}
[i**2 for i in range(4)]
```

::: {exercise-start}
:label: pi-wallis-ex
:class: dropdown
:::

Compute the decimals of Pi using the Wallis formula:

$$
\pi = 2 \prod_{i=1}^{\infty} \frac{4i^2}{4i^2 - 1}
$$

::: {exercise-end}
:::

::: {solution-start} pi-wallis-ex
:class: dropdown
:::

```{code-cell}
from functools import reduce

pi = 3.14159265358979312

my_pi = 1.0

for i in range(1, 100000):
    my_pi *= 4 * i**2 / (4 * i**2 - 1.0)

my_pi *= 2

print(pi)
print(my_pi)
print(abs(pi - my_pi))
```

```{code-cell}
num = 1
den = 1
for i in range(1, 100000):
    tmp = 4 * i * i
    num *= tmp
    den *= tmp - 1

better_pi = 2 * (num / den)

print(pi)
print(better_pi)
print(abs(pi - better_pi))
print(abs(my_pi - better_pi))
```

Solution in a single line using more advanced constructs (reduce, lambda,
list comprehensions):

```{code-cell}
print(
    2
    * reduce(
        lambda x, y: x * y,
        [float(4 * (i**2)) / ((4 * (i**2)) - 1) for i in range(1, 100000)],
    )
)
```

::: {solution-end}
:::
