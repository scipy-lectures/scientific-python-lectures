Object-oriented programming (OOP)
=================================

Python supports object-oriented programming (OOP). The goals of OOP are:

    * to organize the code, and

    * to re-use code in similar contexts.


Here is a small example: we create a Student **class**, which is an object
gathering several custom functions (**methods**) and variables (**attributes**),
we will be able to use::

    >>> class Student(object):
    ...     def __init__(self, name):
    ...         self.name = name
    ...     def set_age(self, age):
    ...         self.age = age
    ...     def set_major(self, major):
    ...         self.major = major
    ...         
    >>> anna = Student('anna')
    >>> anna.set_age(21)
    >>> anna.set_major('physics')

In the previous example, the Student class has ``__init__``, ``set_age`` and
``set_major`` methods. Its attributes are ``name``, ``age`` and ``major``. We
can call these methods and attributes with the following notation:
``classinstance.method`` or  ``classinstance.attribute``.  The ``__init__``
constructor is a special method we call with: ``MyClass(init parameters if
any)``.

Now, suppose we want to create a new class MasterStudent with the same methods
and attributes as the previous one, but with an additional ``stage`` attribute.
We won't copy the previous class, but **inherit** from it::

    >>> class MasterStudent(Student):
    ...     stage = 'mandatory, from march to june'
    ...     
    >>> james = MasterStudent('james')
    >>> james.stage
    'mandatory, from march to june'
    >>> james.set_age(23)
    >>> james.age
    23

The MasterStudent class inherited from the Student attributes and methods.

Thanks to classes and object oriented programming, we can organize code with
different classes corresponding to different objects we encounter (a Manip
class, an Image class, a Flow class, etc.), with their own methods and
attributes. Then we can use inheritance to consider variations around a base
class and **share** code. Ex : from a Turbulent base class, we can create
derived StokesFlow, NewtonianFlow, PotentialFlow, etc.

