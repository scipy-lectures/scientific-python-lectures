.. |==>| unicode:: U+02794 .. thick rightwards arrow

.. default-role:: py:obj

==========================
Advanced Python Constructs
==========================
:author: Zbigniew Jędrzejewski-Szmek

This chapter is about some features of the Python language which can
be considered advanced --- in the sense that not every language has
them, and also in the sense that they are more useful in more
complicated programs or libraries, but not in the sense of being
particularly specialized, or particularly complicated.

It is important to underline that this chapter is purely about the
language itself --- about features supported through special syntax
complemented by functionality of the Python stdlib, which could not be
implemented through clever external modules.

The process of developing the Python programming language, its syntax,
is unique because it is very transparent, proposed changes are
evaluated from various angles and discussed on public mailing lists,
and the final decision takes into account the balance between the
importance of envisioned use cases, the burden of carrying more
language features, consistency with the rest of the syntax, and
whether the proposed variant is the easiest to read, write, and
understand. This process is formalised in Python Enhancement
Proposals --- PEPs_. As a result, features described in this chapter
were added after it was shown that they indeed solve real problems and
that their use is as simple as possible.

.. _PEPs: http://www.python.org/dev/peps/

.. contents:: Chapters contents
   :local:
   :depth: 4



Iterators, generator expressions and generators
===============================================

Iterators
^^^^^^^^^

.. sidebar:: Simplicity

   Duplication of effort is wasteful, and replacing the various
   home-grown approaches with a standard feature usually ends up
   making things more readable, and interoperable as well.

                 *Guido van Rossum* --- `Adding Optional Static Typing to Python`_

.. _`Adding Optional Static Typing to Python`:
   http://www.artima.com/weblogs/viewpost.jsp?thread=86641


An iterator is an object adhering to the `iterator protocol`_
--- basically this means that it has a `next <iterator.next>` method,
which, when called, returns the next item in the sequence, and when
there's nothing to return, raises the
`StopIteration <exceptions.StopIteration>` exception.

.. _`iterator protocol`: http://docs.python.org/dev/library/stdtypes.html#iterator-types

An iterator object allows to loop just once. It
holds the state (position) of a single iteration, or from the other
side, each loop over a sequence requires a single iterator
object. This means that we can iterate over the same sequence more
than once concurrently. Separating the iteration logic from the
sequence allows us to have more than one way of iteration.

Calling the `__iter__ <object.__iter__>` method on a container to
create an iterator object is the most straightforward way to get hold
of an iterator. The `iter` function does that for us, saving a few
keystrokes.

>>> nums = [1,2,3]      # note that ... varies: these are different objects
>>> iter(nums)                           # doctest: +ELLIPSIS
<listiterator object at ...>
>>> nums.__iter__()                      # doctest: +ELLIPSIS
<listiterator object at ...>
>>> nums.__reversed__()                  # doctest: +ELLIPSIS
<listreverseiterator object at ...>

>>> it = iter(nums)
>>> next(it)            # next(obj) simply calls obj.next()
1
>>> it.next()
2
>>> next(it)
3
>>> next(it)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration

When used in a loop, `StopIteration <exceptions.StopIteration>` is
swallowed and causes the loop to finish. But with explicit invocation,
we can see that once the iterator is exhausted, accessing it raises an
exception.

Using the :compound:`for..in <for>` loop also uses the ``__iter__``
method. This allows us to transparently start the iteration over a
sequence. But if we already have the iterator, we want to be able to
use it in an ``for`` loop in the same way. In order to achieve this,
iterators in addition to ``next`` are also required to have a method
called ``__iter__`` which returns the iterator (``self``).

Support for iteration is pervasive in Python:
all sequences and unordered containers in the standard library allow
this. The concept is also stretched to other things:
e.g. ``file`` objects support iteration over lines.

>>> f = open('/etc/fstab')
>>> f is f.__iter__()
True

The ``file`` is an iterator itself and it's ``__iter__`` method
doesn't create a separate object: only a single thread of sequential
access is allowed.

Generator expressions
^^^^^^^^^^^^^^^^^^^^^

A second way in which iterator objects are created is through
**generator expressions**, the basis for **list comprehensions**. To
increase clarity, a generator expression must always be enclosed in
parentheses or an expression. If round parentheses are used, then a
generator iterator is created.  If rectangular parentheses are used,
the process is short-circuited and we get a ``list``. ::

    >>> (i for i in nums)                    # doctest: +ELLIPSIS
    <generator object <genexpr> at 0x...>
    >>> [i for i in nums]
    [1, 2, 3]
    >>> list(i for i in nums)
    [1, 2, 3]

In Python 2.7 and 3.x the list comprehension syntax was extended to
**dictionary and set comprehensions**.
A ``set`` is created when the generator expression is enclosed in curly
braces. A ``dict`` is created when the generator expression contains
"pairs" of the form ``key:value``::

    >>> {i for i in range(3)}   # doctest: +SKIP
    set([0, 1, 2])
    >>> {i:i**2 for i in range(3)}   # doctest: +SKIP
    {0: 0, 1: 1, 2: 4}

If you are stuck at some previous Python version, the syntax is only a
bit worse: ::

    >>> set(i for i in 'abc')
    set(['a', 'c', 'b'])
    >>> dict((i, ord(i)) for i in 'abc')
    {'a': 97, 'c': 99, 'b': 98}

Generator expression are fairly simple, not much to say here. Only one
*gotcha* should be mentioned: in old Pythons the index variable
(``i``) would leak, and in versions >= 3 this is fixed.

Generators
^^^^^^^^^^

.. sidebar:: Generators

  A generator is a function that produces a
  sequence of results instead of a single value.

          *David Beazley* --- `A Curious Course on Coroutines and Concurrency`_

.. _`A Curious Course on Coroutines and Concurrency`:
   http://www.dabeaz.com/coroutines/

A third way to create iterator objects is to call a generator function.
A **generator** is a function containing the keyword :simple:`yield`. It must be
noted that the mere presence of this keyword completely changes the
nature of the function: this ``yield`` statement doesn't have to be
invoked, or even reachable, but causes the function to be marked as a
generator. When a normal function is called, the instructions
contained in the body start to be executed. When a generator is
called, the execution stops before the first instruction in the body.
An invocation of a generator function creates a generator object,
adhering to the iterator protocol. As with normal function
invocations, concurrent and recursive invocations are allowed.

When ``next`` is called, the function is executed until the first ``yield``.
Each encountered ``yield`` statement gives a value becomes the return
value of ``next``. After executing the ``yield`` statement, the
execution of this function is suspended. ::

    >>> def f():
    ...   yield 1
    ...   yield 2
    >>> f()                                   # doctest: +ELLIPSIS
    <generator object f at 0x...>
    >>> gen = f()
    >>> gen.next()
    1
    >>> gen.next()
    2
    >>> gen.next()
    Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
    StopIteration

Let's go over the life of the single invocation of the generator
function. ::

    >>> def f():
    ...   print("-- start --")
    ...   yield 3
    ...   print("-- middle --")
    ...   yield 4
    ...   print("-- finished --")
    >>> gen = f()
    >>> next(gen)
    -- start --
    3
    >>> next(gen)
    -- middle --
    4
    >>> next(gen)                            # doctest: +SKIP
    -- finished --
    Traceback (most recent call last):
     ...
    StopIteration

Contrary to a normal function, where executing ``f()`` would
immediately cause the first ``print`` to be executed, ``gen`` is
assigned without executing any statements in the function body. Only
when ``gen.next()`` is invoked by ``next``, the statements up to
the first ``yield`` are executed. The second ``next`` prints
``-- middle --`` and execution halts on the second ``yield``.  The third
``next`` prints ``-- finished --`` and falls of the end of the
function. Since no ``yield`` was reached, an exception is raised.

What happens with the function after a yield, when the control passes
to the caller? The state of each generator is stored in the generator
object. From the point of view of the generator function, is looks
almost as if it was running in a separate thread, but this is just an
illusion: execution is strictly single-threaded, but the interpreter
keeps and restores the state in between the requests for the next value.

Why are generators useful? As noted in the parts about iterators, a
generator function is just a different way to create an iterator
object. Everything that can be done with ``yield`` statements, could
also be done with ``next`` methods. Nevertheless, using a
function and having the interpreter perform its magic to create an
iterator has advantages. A function can be much shorter
than the definition of a class with the required ``next`` and
``__iter__`` methods. What is more important, it is easier for the author
of the generator to understand the state which is kept in local
variables, as opposed to instance attributes, which have to be
used to pass data between consecutive invocations of ``next`` on
an iterator object.

A broader question is why are iterators useful? When an iterator is
used to power a loop, the loop becomes very simple. The code to
initialise the state, to decide if the loop is finished, and to find
the next value is extracted into a separate place. This highlights the
body of the loop --- the interesting part. In addition, it is possible
to reuse the iterator code in other places.

Bidirectional communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``yield`` statement causes a value to be passed to the
caller. This is the reason for the introduction of generators
by :pep:`255` (implemented in Python 2.2).  But communication in the
reverse direction is also useful. One obvious way would be some
external state, either a global variable or a shared mutable
object. Direct communication is possible thanks to :pep:`342`
(implemented in 2.5). It is achieved by turning the previously boring
``yield`` statement into an expression. When the generator resumes
execution after a ``yield`` statement, the caller can call a method on
the generator object to either pass a value **into** the generator,
which then is returned by the ``yield`` statement, or a
different method to inject an exception into the generator.

The first of the new methods is `send(value) <generator.send>`, which
is similar to `next() <generator.next>`, but passes ``value`` into
the generator to be used for the value of the ``yield`` expression. In
fact, ``g.next()`` and ``g.send(None)`` are equivalent.

The second of the new methods is
`throw(type, value=None, traceback=None) <generator.throw>`
which is equivalent to::

  raise type, value, traceback

at the point of the ``yield`` statement.

Unlike :simple:`raise` (which immediately raises an exception from the
current execution point), ``throw()`` first resumes the generator, and
only then raises the exception.  The word throw was picked because
it is suggestive of putting the exception in another location, and is
associated with exceptions in other languages.

What happens when an exception is raised inside the generator? It can
be either raised explicitly or when executing some statements or it
can be injected at the point of a ``yield`` statement by means of the
``throw()`` method. In either case, such an exception propagates in the
standard manner: it can be intercepted by an ``except`` or ``finally``
clause, or otherwise it causes the execution of the generator function
to be aborted and propagates in the caller.

For completeness' sake, it's worth mentioning that generator iterators
also have a `close() <generator.close>` method, which can be used to
force a generator that would otherwise be able to provide more values
to finish immediately. It allows the generator `__del__ <object.__del__>`
method to destroy objects holding the state of generator.

Let's define a generator which just prints what is passed in through
send and throw. ::

    >>> import itertools
    >>> def g():
    ...     print '--start--'
    ...     for i in itertools.count():
    ...         print '--yielding %i--' % i
    ...         try:
    ...             ans = yield i
    ...         except GeneratorExit:
    ...             print '--closing--'
    ...             raise
    ...         except Exception as e:
    ...             print '--yield raised %r--' % e
    ...         else:
    ...             print '--yield returned %s--' % ans

    >>> it = g()
    >>> next(it)
    --start--
    --yielding 0--
    0
    >>> it.send(11)
    --yield returned 11--
    --yielding 1--
    1
    >>> it.throw(IndexError)
    --yield raised IndexError()--
    --yielding 2--
    2
    >>> it.close()
    --closing--

.. note:: ``next`` or ``__next__``?

  In Python 2.x, the iterator method to retrieve the next value is
  called `next <iterator.next>`. It is invoked implicitly through the
  global function `next`, which means that it should be called ``__next__``.
  Just like the global function `iter` calls `__iter__ <iterator.__iter__>`.
  This inconsistency is corrected in Python 3.x, where ``it.next``
  becomes ``it.__next__``.  For other generator methods --- ``send``
  and ``throw`` --- the situation is more complicated, because they
  are not called implicitly by the interpreter. Nevertheless, there's
  a proposed syntax extension to allow ``continue`` to take an
  argument which will be passed to `send <generator.send>` of the
  loop's iterator. If this extension is accepted, it's likely that
  ``gen.send`` will become ``gen.__send__``. The last of generator
  methods, `close <generator.close>`, is pretty obviously named
  incorrectly, because it is already invoked implicitly.

Chaining generators
^^^^^^^^^^^^^^^^^^^

.. note::

  This is a preview of :pep:`380` (not yet implemented, but accepted
  for Python 3.3).

Let's say we are writing a generator and we want to yield a number of
values generated by a second generator, a **subgenerator**.
If yielding of values is the only concern, this can be performed
without much difficulty using a loop such as

.. code-block:: python

  subgen = some_other_generator()
  for v in subgen:
      yield v

However, if the subgenerator is to interact properly with the caller
in the case of calls to ``send()``, ``throw()`` and ``close()``,
things become considerably more difficult. The ``yield`` statement has
to be guarded by a :compound:`try..except..finally <try>` structure
similar to the one defined in the previous section to "debug" the
generator function.  Such code is provided in :pep:`380#id13`, here it
suffices to say that new syntax to properly yield from a subgenerator
is being introduced in Python 3.3:

.. code-block:: python

   yield from some_other_generator()

This behaves like the explicit loop above, repeatedly yielding values
from ``some_other_generator`` until it is exhausted, but also forwards
``send``, ``throw`` and ``close`` to the subgenerator.

Decorators
==========

.. sidebar:: Summary

   This amazing feature appeared in the language almost apologetically
   and with concern that it might not be that useful.

                   *Bruce Eckel* --- An Introduction to Python Decorators

.. documentation error:
.. The result must be a class object, which is then bound to the class name.
.. file:///usr/share/doc/python2.7/html/reference/compound_stmts.html
.. >>> def deco(cls):return None
.. ...
.. >>> @deco
.. ... class A: pass
.. ...
.. >>> A
.. >>> type(A)
.. <class 'NoneType'>
.. >>> print(A)
.. None

Since a function or a class are objects, they can be passed
around. Since they are mutable objects, they can be modified.  The act
of altering a function or class object after it has been constructed
but before is is bound to its name is called decorating.

There are two things hiding behind the name "decorator" --- one is the
function which does the work of decorating, i.e. performs the real
work, and the other one is the expression adhering to the decorator
syntax, i.e. an at-symbol and the name of the decorating function.

Function can be decorated by using the decorator syntax for
functions::

    @decorator             # ②
    def function():        # ①
        pass

- A function is defined in the standard way. ①
- An expression starting with ``@`` placed before the function
  definition is the decorator ②. The part after ``@`` must be a simple
  expression, usually this is just the name of a function or class. This
  part is evaluated first, and after the function defined below is
  ready, the decorator is called with the newly defined function object
  as the single argument. The value returned by the decorator is
  attached to the original name of the function.

Decorators can be applied to functions and to classes. For
classes the semantics are identical --- the original class definition
is used as an argument to call the decorator and whatever is returned
is assigned under the original name.

Before the decorator syntax was implemented (:pep:`318`), it was
possible to achieve the same effect by assigning the function or class
object to a temporary variable and then invoking the decorator
explicitly and then assigning the return value to the name of the
function. This sound like more typing, and it is, and also the name of
the decorated function dubbling as a temporary variable must be used
at least three times, which is prone to errors. Nevertheless, the
example above is equivalent to::

    def function():                  # ①
        pass
    function = decorator(function)   # ②

Decorators can be stacked --- the order of application is
bottom-to-top, or inside-out. The semantics are such that the originally
defined function is used as an argument for the first decorator,
whatever is returned by the first decorator is used as an argument for
the second decorator, ..., and whatever is returned by the last
decorator is attached under the name of the original function.

The decorator syntax was chosen for its readability. Since the
decorator is specified before the header of the function, it is
obvious that its is not a part of the function body and its clear that
it can only operate on the whole function. Because the expression is
prefixed with ``@`` is stands out and is hard to miss ("in your face",
according to the PEP :) ). When more than one decorator is applied,
each one is placed on a separate line in an easy to read way.


Replacing or tweaking the original object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Decorators can either return the same function or class object or they
can return a completely different object. In the first case, the
decorator can exploit the fact that function and class objects are
mutable and add attributes, e.g. add a docstring to a class. A
decorator might do something useful even without modifying the object,
for example register the decorated class in a global registry. In the
second case, virtually anything is possible: when a something
different is substituted for the original function or class, the new
object can be completely different. Nevertheless, such behaviour is
not the purpose of decorators: they are intended to tweak the
decorated object, not do something unpredictable. Therefore, when a
function is "decorated" by replacing it with a different function, the
new function usually calls the original function, after doing some
preparatory work. Likewise, when a class is "decorated" by replacing
if with a new class, the new class is usually derived from the
original class. When the purpose of the decorator is to do something
"every time", like to log every call to a decorated function, only the
second type of decorators can be used. On the other hand, if the first
type is sufficient, it is better to use it, because it is simpler.

Decorators implemented as classes and as functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The only *requirement* on decorators is that they can be called with a
single argument. This means that decorators can be implemented as
normal functions, or as classes with a `__call__ <object.__call__>`
method, or in theory, even as lambda functions.

Let's compare the function and class approaches. The decorator
expression (the part after ``@``) can be either just a name, or a
call. The bare-name approach is nice (less to type, looks cleaner,
etc.), but is only possible when no arguments are needed to customise
the decorator. Decorators written as functions can be used in those
two cases:

>>> def simple_decorator(function):
...   print "doing decoration"
...   return function
>>> @simple_decorator
... def function():
...   print "inside function"
doing decoration
>>> function()
inside function

>>> def decorator_with_arguments(arg):
...   print "defining the decorator"
...   def _decorator(function):
...       # in this inner function, arg is available too
...       print "doing decoration,", arg
...       return function
...   return _decorator
>>> @decorator_with_arguments("abc")
... def function():
...   print "inside function"
defining the decorator
doing decoration, abc
>>> function()
inside function

The two trivial decorators above fall into the category of decorators
which return the original function. If they were to return a new
function, an extra level of nestedness would be required.
In the worst case, three levels of nested functions.

>>> def replacing_decorator_with_args(arg):
...   print "defining the decorator"
...   def _decorator(function):
...       # in this inner function, arg is available too
...       print "doing decoration,", arg
...       def _wrapper(*args, **kwargs):
...           print "inside wrapper,", args, kwargs
...           return function(*args, **kwargs)
...       return _wrapper
...   return _decorator
>>> @replacing_decorator_with_args("abc")
... def function(*args, **kwargs):
...     print "inside function,", args, kwargs
...     return 14
defining the decorator
doing decoration, abc
>>> function(11, 12)
inside wrapper, (11, 12) {}
inside function, (11, 12) {}
14

The ``_wrapper`` function is defined to accept all positional and
keyword arguments. In general we cannot know what arguments the
decorated function is supposed to accept, so the wrapper function
just passes everything to the wrapped function. One unfortunate
consequence is that the apparent argument list is misleading.

Compared to decorators defined as functions, complex decorators
defined as classes are simpler.  When an object is created, the
`__init__ <object.__init__>` method is only allowed to return `None`,
and the type of the created object cannot be changed. This means that
when a decorator is defined as a class, it doesn't make much sense to
use the argument-less form: the final decorated object would just be
an instance of the decorating class, returned by the constructor call,
which is not very useful. Therefore it's enough to discuss class-based
decorators where arguments are given in the decorator expression and
the decorator ``__init__`` method is used for decorator construction.

>>> class decorator_class(object):
...   def __init__(self, arg):
...       # this method is called in the decorator expression
...       print "in decorator init,", arg
...       self.arg = arg
...   def __call__(self, function):
...       # this method is called to do the job
...       print "in decorator call,", self.arg
...       return function
>>> deco_instance = decorator_class('foo')
in decorator init, foo
>>> @deco_instance
... def function(*args, **kwargs):
...   print "in function,", args, kwargs
in decorator call, foo
>>> function()
in function, () {}

Contrary to normal rules (:PEP:`8`) decorators written as classes
behave more like functions and therefore their name often starts with a
lowercase letter.

In reality, it doesn't make much sense to create a new class just to
have a decorator which returns the original function. Objects are
supposed to hold state, and such decorators are more useful when the
decorator returns a new object.

>>> class replacing_decorator_class(object):
...   def __init__(self, arg):
...       # this method is called in the decorator expression
...       print "in decorator init,", arg
...       self.arg = arg
...   def __call__(self, function):
...       # this method is called to do the job
...       print "in decorator call,", self.arg
...       self.function = function
...       return self._wrapper
...   def _wrapper(self, *args, **kwargs):
...       print "in the wrapper,", args, kwargs
...       return self.function(*args, **kwargs)
>>> deco_instance = replacing_decorator_class('foo')
in decorator init, foo
>>> @deco_instance
... def function(*args, **kwargs):
...   print "in function,", args, kwargs
in decorator call, foo
>>> function(11, 12)
in the wrapper, (11, 12) {}
in function, (11, 12) {}

A decorator like this can do pretty much anything, since it can modify
the original function object and mangle the arguments, call the
original function or not, and afterwards mangle the return value.

Copying the docstring and other attributes of the original function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a new function is returned by the decorator to replace the
original function, an unfortunate consequence is that the original
function name, the original docstring, the original argument list are
lost. Those attributes of the original function can partially be "transplanted"
to the new function by setting ``__doc__`` (the docstring), ``__module__``
and ``__name__`` (the full name of the function), and
``__annotations__`` (extra information about arguments and the return
value of the function available in Python 3). This can be done
automatically by using `functools.update_wrapper`.

.. sidebar:: `functools.update_wrapper(wrapper, wrapped) <functools.update_wrapper>`

   "Update a wrapper function to look like the wrapped function."

>>> import functools
>>> def better_replacing_decorator_with_args(arg):
...   print "defining the decorator"
...   def _decorator(function):
...       print "doing decoration,", arg
...       def _wrapper(*args, **kwargs):
...           print "inside wrapper,", args, kwargs
...           return function(*args, **kwargs)
...       return functools.update_wrapper(_wrapper, function)
...   return _decorator
>>> @better_replacing_decorator_with_args("abc")
... def function():
...     "extensive documentation"
...     print "inside function"
...     return 14
defining the decorator
doing decoration, abc
>>> function                           # doctest: +ELLIPSIS
<function function at 0x...>
>>> print function.__doc__
extensive documentation

One important thing is missing from the list of attributes which can
be copied to the replacement function: the argument list. The default
values for arguments can be modified through the ``__defaults__``,
``__kwdefaults__`` attributes, but unfortunately the argument list
itself cannot be set as an attribute. This means that
``help(function)`` will display a useless argument list which will be
confusing for the user of the function. An effective but ugly way
around this problem is to create the wrapper dynamically, using
``eval``. This can be automated by using the external ``decorator``
module. It provides support the ``decorator`` decorator, which takes a
wrapper and turns it into a decorator which preserves the function
signature.

To sum things up, decorators should always use ``functools.update_wrapper``
or some other means of copying function attributes.

Examples in the standard library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, it should be mentioned that there's a number of useful
decorators available in the standard library. There are three decorators
which really form a part of the language:

- `classmethod` causes a method to become a "class method",
  which means that it can be invoked without creating an instance of
  the class. When a normal method is invoked, the interpreter inserts
  the instance object as the first positional parameter,
  ``self``. When a class method is invoked, the class itself is given
  as the first parameter, often called ``cls``.

  Class methods are still accessible through the class' namespace, so
  they don't pollute the module's namespace. Class methods can be used
  to provide alternative constructors::

    class Array(object):
        def __init__(self, data):
	    self.data = data

        @classmethod
        def fromfile(cls, file):
            data = numpy.load(file)
            return cls(data)

  This is cleaner then using a multitude of flags to ``__init__``.

- `staticmethod` is applied to methods to make them "static",
  i.e. basically a normal function, but accessible through the class
  namespace. This can be useful when the function is only needed
  inside this class (its name would then be prefixed with ``_``), or when we
  want the user to think of the method as connected to the class,
  despite an implementation which doesn't require this.

- `property` is the pythonic answer to the problem of getters
  and setters. A method decorated with ``property`` becomes a getter
  which is automatically called on attribute access.

  >>> class A(object):
  ...   @property
  ...   def a(self):
  ...     "an important attribute"
  ...     return "a value"
  >>> A.a                                   # doctest: +ELLIPSIS
  <property object at 0x...>
  >>> A().a
  'a value'

  In this example, ``A.a`` is an read-only attribute. It is also
  documented: ``help(A)`` includes the docstring for attribute ``a``
  taken from the getter method. Defining ``a`` as a property allows it
  to be a calculated on the fly, and has the side effect of making it
  read-only, because no setter is defined.

  To have a setter and a getter, two methods are required,
  obviously. Since Python 2.6 the following syntax is preferred::

    class Rectangle(object):
        def __init__(self, edge):
            self.edge = edge

        @property
        def area(self):
            """Computed area.

            Setting this updates the edge length to the proper value.
            """
            return self.edge**2

        @area.setter
        def area(self, area):
            self.edge = area ** 0.5

  The way that this works, is that the ``property`` decorator replaces
  the getter method with a property object. This object in turn has
  three methods, ``getter``, ``setter``, and ``deleter``, which can be
  used as decorators. Their job is to set the getter, setter and
  deleter of the property object (stored as attributes ``fget``,
  ``fset``, and ``fdel``). The getter can be set like in the example
  above, when creating the object. When defining the setter, we
  already have the property object under ``area``, and we add the
  setter to it by using the ``setter`` method. All this happens when
  we are creating the class.

  Afterwards, when an instance of the class has been created, the
  property object is special. When the interpreter executes attribute
  access, assignment, or deletion, the job is delegated to the methods
  of the property object.

  To make everything crystal clear, let's define a "debug" example::

    >>> class D(object):
    ...    @property
    ...    def a(self):
    ...      print "getting", 1
    ...      return 1
    ...    @a.setter
    ...    def a(self, value):
    ...      print "setting", value
    ...    @a.deleter
    ...    def a(self):
    ...      print "deleting"
    >>> D.a                                    # doctest: +ELLIPSIS
    <property object at 0x...>
    >>> D.a.fget                               # doctest: +ELLIPSIS
    <function a at 0x...>
    >>> D.a.fset                               # doctest: +ELLIPSIS
    <function a at 0x...>
    >>> D.a.fdel                               # doctest: +ELLIPSIS
    <function a at 0x...>
    >>> d = D()               # ... varies, this is not the same `a` function
    >>> d.a
    getting 1
    1
    >>> d.a = 2
    setting 2
    >>> del d.a
    deleting
    >>> d.a
    getting 1
    1

  Properties are a bit of a stretch for the decorator syntax. One of the
  premises of the decorator syntax --- that the name is not duplicated
  --- is violated, but nothing better has been invented so far. It is
  just good style to use the same name for the getter, setter, and
  deleter methods.

  .. property documentation mentions that this only works for
     old-style classes, but this seems to be an error.

Some newer examples include:

- `functools.lru_cache` memoizes an arbitrary function
  maintaining a limited cache of arguments:answer pairs (Python 3.2)

- `functools.total_ordering` is a class decorator which fills in
  missing ordering methods
  (`__lt__ <object.__lt__>`, `__gt__ <object.__gt__>`,
  `__le__ <object.__le__>`, ...)
  based on a single available one (Python 2.7).


..
  - `packaging.pypi.simple.socket_timeout` (in Python 3.3) adds
  a socket timeout when retrieving data through a socket.


Deprecation of functions
^^^^^^^^^^^^^^^^^^^^^^^^

Let's say we want to print a deprecation warning on stderr on the
first invocation of a function we don't like anymore. If we don't want
to modify the function, we can use a decorator::

  class deprecated(object):
      """Print a deprecation warning once on first use of the function.

      >>> @deprecated()                    # doctest: +SKIP
      ... def f():
      ...     pass
      >>> f()                              # doctest: +SKIP
      f is deprecated
      """
      def __call__(self, func):
	  self.func = func
	  self.count = 0
	  return self._wrapper
      def _wrapper(self, *args, **kwargs):
	  self.count += 1
	  if self.count == 1:
	      print self.func.__name__, 'is deprecated'
	  return self.func(*args, **kwargs)

.. TODO: use update_wrapper here

It can also be implemented as a function::

  def deprecated(func):
      """Print a deprecation warning once on first use of the function.

      >>> @deprecated                      # doctest: +SKIP
      ... def f():
      ...     pass
      >>> f()                              # doctest: +SKIP
      f is deprecated
      """
      count = [0]
      def wrapper(*args, **kwargs):
          count[0] += 1
          if count[0] == 1:
              print func.__name__, 'is deprecated'
          return func(*args, **kwargs)
      return wrapper

A ``while``-loop removing decorator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's say we have function which returns a lists of things, and this
list created by running a loop. If we don't know how many objects will
be needed, the standard way to do this is something like::

  def find_answers():
      answers = []
      while True:
	  ans = look_for_next_answer()
	  if ans is None:
	      break
	  answers.append(ans)
      return answers

This is fine, as long as the body of the loop is fairly compact. Once
it becomes more complicated, as often happens in real code, this
becomes pretty unreadable. We could simplify this by using ``yield``
statements, but then the user would have to explicitly call
``list(find_answers())``.

We can define a decorator which constructs the list for us::

  def vectorized(generator_func):
      def wrapper(*args, **kwargs):
	  return list(generator_func(*args, **kwargs))
      return functools.update_wrapper(wrapper, generator_func)

Our function then becomes::

  @vectorized
  def find_answers():
      while True:
	  ans = look_for_next_answer()
	  if ans is None:
	      break
	  yield ans

A plugin registration system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a class decorator which doesn't modify the class, but just
puts it in a global registry. It falls into the category of decorators
returning the original object::

  class WordProcessor(object):
      PLUGINS = []
      def process(self, text):
          for plugin in self.PLUGINS:
              text = plugin().cleanup(text)
          return text

      @classmethod
      def plugin(cls, plugin):
          cls.PLUGINS.append(plugin)

  @WordProcessor.plugin
  class CleanMdashesExtension(object):
      def cleanup(self, text):
          return text.replace('&mdash;', u'\N{em dash}')

Here we use a decorator to decentralise the registration of
plugins. We call our decorator with a noun, instead of a verb, because
we use it to declare that our class is a plugin for
``WordProcessor``. Method ``plugin`` simply appends the class to the
list of plugins.

A word about the plugin itself: it replaces HTML entity for em-dash
with a real Unicode em-dash character. It exploits the `unicode
literal notation`_ to insert a character by using its name in the
unicode database ("EM DASH"). If the Unicode character was inserted
directly, it would be impossible to distinguish it from an en-dash in
the source of a program.

.. _`unicode literal notation`:
   http://docs.python.org/2.7/reference/lexical_analysis.html#string-literals

More examples and reading
^^^^^^^^^^^^^^^^^^^^^^^^^

* :pep:`318` (function and method decorator syntax)
* :pep:`3129` (class decorator syntax)
* http://wiki.python.org/moin/PythonDecoratorLibrary
* http://docs.python.org/dev/library/functools.html
* http://pypi.python.org/pypi/decorator
* Bruce Eckel

  - `Decorators I`_: Introduction to Python Decorators
  - `Python Decorators II`_: Decorator Arguments
  - `Python Decorators III`_: A Decorator-Based Build System

.. _`Decorators I`: http://www.artima.com/weblogs/viewpost.jsp?thread=240808
.. _`Python Decorators II`: http://www.artima.com/weblogs/viewpost.jsp?thread=240845
.. _`Python Decorators III`: http://www.artima.com/weblogs/viewpost.jsp?thread=241209


Context managers
================

A context manager is an object with `__enter__ <object.__enter__>` and
`__exit__ <object.__exit__>` methods which can be used in the :compound:`with`
statement::

  with manager as var:
      do_something(var)

is in the simplest case
equivalent to ::

  var = manager.__enter__()
  try:
      do_something(var)
  finally:
      manager.__exit__()

In other words, the context manager protocol defined in :pep:`343`
permits the extraction of the boring part of a
:compound:`try..except..finally <try>` structure into a separate class
leaving only the interesting ``do_something`` block.

1. The `__enter__ <object.__enter__>` method is called first.  It can
   return a value which will be assigned to ``var``.
   The ``as``-part is optional: if it isn't present, the value
   returned by ``__enter__`` is simply ignored.
2. The block of code underneath ``with`` is executed.  Just like with
   ``try`` clauses, it can either execute successfully to the end, or
   it can :simple:`break`, :simple:`continue`` or :simple:`return`, or
   it can throw an exception. Either way, after the block is finished,
   the `__exit__ <object.__exit__>` method is called.
   If an exception was thrown, the information about the exception is
   passed to ``__exit__``, which is described below in the next
   subsection. In the normal case, exceptions can be ignored, just
   like in a ``finally`` clause, and will be rethrown after
   ``__exit__`` is finished.

Let's say we want to make sure that a file is closed immediately after
we are done writing to it::

  >>> class closing(object):
  ...   def __init__(self, obj):
  ...     self.obj = obj
  ...   def __enter__(self):
  ...     return self.obj
  ...   def __exit__(self, *args):
  ...     self.obj.close()
  >>> with closing(open('/tmp/file', 'w')) as f:
  ...   f.write('the contents\n')

Here we have made sure that the ``f.close()`` is called when the
``with`` block is exited. Since closing files is such a common
operation, the support for this is already present in the ``file``
class. It has an ``__exit__`` method which calls ``close`` and can be
used as a context manager itself::

  >>> with open('/tmp/file', 'a') as f:
  ...   f.write('more contents\n')

The common use for ``try..finally`` is releasing resources. Various
different cases are implemented similarly: in the ``__enter__``
phase the resource is acquired, in the ``__exit__`` phase it is
released, and the exception, if thrown, is propagated. As with files,
there's often a natural operation to perform after the object has been
used and it is most convenient to have the support built in. With each
release, Python provides support in more places:

* all file-like objects:

  - `file` |==>| automatically closed
  - `fileinput`, `tempfile` (py >= 3.2)
  - `bz2.BZ2File`, `gzip.GzipFile`,
    `tarfile.TarFile`, `zipfile.ZipFile`
  - `ftplib`, `nntplib` |==>| close connection (py >= 3.2 or 3.3)
* locks

  - `multiprocessing.RLock` |==>| lock and unlock
  - `multiprocessing.Semaphore`
  - `memoryview` |==>| automatically release (py >= 3.2 and 2.7)
* `decimal.localcontext` |==>| modify precision of computations temporarily
* `_winreg.PyHKEY <_winreg.OpenKey>` |==>| open and close hive key
* `warnings.catch_warnings` |==>| kill warnings temporarily
* `contextlib.closing` |==>| the same as the example above, call ``close``
* parallel programming

  - `concurrent.futures.ThreadPoolExecutor` |==>| invoke in parallel then kill thread pool (py >= 3.2)
  - `concurrent.futures.ProcessPoolExecutor` |==>| invoke in parallel then kill process pool (py >= 3.2)
  - `nogil` |==>| solve the GIL problem temporarily (cython only :( )


Catching exceptions
^^^^^^^^^^^^^^^^^^^

When an exception is thrown in the ``with``-block, it is passed as
arguments to ``__exit__``. Three arguments are used, the same as
returned by :py:func:`sys.exc_info`: type, value, traceback. When no
exception is thrown, ``None`` is used for all three arguments.  The
context manager can "swallow" the exception by returning a true value
from ``__exit__``. Exceptions can be easily ignored, because if
``__exit__`` doesn't use ``return`` and just falls of the end,
``None`` is returned, a false value, and therefore the exception is
rethrown after ``__exit__`` is finished.

The ability to catch exceptions opens interesting possibilities. A
classic example comes from unit-tests --- we want to make sure that
some code throws the right kind of exception::

  class assert_raises(object):
      # based on pytest and unittest.TestCase
      def __init__(self, type):
          self.type = type
      def __enter__(self):
          pass
      def __exit__(self, type, value, traceback):
          if type is None:
              raise AssertionError('exception expected')
          if issubclass(type, self.type):
              return True # swallow the expected exception
          raise AssertionError('wrong exception type')

  with assert_raises(KeyError):
      {}['foo']

Using generators to define context managers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When discussing generators_, it was said that we prefer generators to
iterators implemented as classes because they are shorter, sweeter,
and the state is stored as local, not instance, variables. On the
other hand, as described in `Bidirectional communication`_, the flow
of data between the generator and its caller can be bidirectional.
This includes exceptions, which can be thrown into the
generator. We would like to implement context managers as special
generator functions. In fact, the generator protocol was designed to
support this use case.

.. code-block:: python

  @contextlib.contextmanager
  def some_generator(<arguments>):
      <setup>
      try:
	  yield <value>
      finally:
	  <cleanup>

The `contextlib.contextmanager` helper takes a generator and turns it
into a context manager. The generator has to obey some rules which are
enforced by the wrapper function --- most importantly it must
``yield`` exactly once. The part before the ``yield`` is executed from
``__enter__``, the block of code protected by the context manager is
executed when the generator is suspended in ``yield``, and the rest is
executed in ``__exit__``. If an exception is thrown, the interpreter
hands it to the wrapper through ``__exit__`` arguments, and the
wrapper function then throws it at the point of the ``yield``
statement. Through the use of generators, the context manager is
shorter and simpler.

Let's rewrite the ``closing`` example as a generator::

  @contextlib.contextmanager
  def closing(obj):
      try:
	  yield obj
      finally:
	  obj.close()

Let's rewrite the ``assert_raises`` example as a generator::

  @contextlib.contextmanager
  def assert_raises(type):
      try:
	  yield
      except type:
	  return
      except Exception as value:
	  raise AssertionError('wrong exception type')
      else:
	  raise AssertionError('exception expected')

Here we use a decorator to turn generator functions into context managers!
