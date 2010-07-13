.. sectionauthor:: Pauli Virtanen

Abstract
========

Numpy is at the base of Python's scientific stack of tools.
Its purpose is simple: implementing efficient operations on
many items in a block of memory.  Understanding how it works
in detail helps in making efficient use of its flexibility,
taking useful shortcuts, and in building new work based on
it.

This tutorial aims to cover:

- Anatomy of Numpy arrays, and its consequences. Tips and
  tricks.

- Universal functions: what, why, and what to do if you want
  a new one.

- Integration with other tools: Numpy offers several ways to
  wrap any data in an ndarray, without unnecessary copies.

- Recently added features, and what's in them for me: PEP
  3118 buffers, generalized ufuncs, ...

