Assignment operator
===================

`Python library reference
<http://docs.python.org/reference/simple_stmts.html#assignment-statements>`_
says:

  Assignment statements are used to (re)bind names to values and to
  modify attributes or items of mutable objects.

In short, it works as follows (simple assignment):

#. an expression on the right hand side is evaluated, the corresponding
   object is created/obtained
#. a **name** on the left hand side is assigned, or bound, to the
   r.h.s. object

Things to note:

* a single object can have several names bound to it:

    .. sourcecode:: ipython

        In [1]: a = [1, 2, 3]
        In [2]: b = a
        In [3]: a
        Out[3]: [1, 2, 3]
        In [4]: b
        Out[4]: [1, 2, 3]
        In [5]: a is b
        Out[5]: True
	In [6]: b[1] = 'hi!'
	In [7]: a
	Out[7]: [1, 'hi!', 3]

* to change a list *in place*, use indexing/slices:

    .. sourcecode:: ipython

        In [1]: a = [1, 2, 3]
        In [3]: a
        Out[3]: [1, 2, 3]
        In [4]: a = ['a', 'b', 'c'] # Creates another object.
        In [5]: a
        Out[5]: ['a', 'b', 'c']
        In [6]: id(a)
        Out[6]: 138641676
        In [7]: a[:] = [1, 2, 3] # Modifies object in place.
        In [8]: a
        Out[8]: [1, 2, 3]
        In [9]: id(a)
        Out[9]: 138641676 # Same as in Out[6], yours will differ...

* the key concept here is **mutable vs. immutable**

    * mutable objects can be changed in place
    * immutable objects cannot be modified once created

A very good and detailed explanation of the above issues can be found
in David M. Beazley's article `Types and Objects in Python
<http://www.informit.com/articles/article.aspx?p=453682>`_.
