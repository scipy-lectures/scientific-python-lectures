=================
Optimizing code
=================

.. sidebar:: Donald Knuth

   *“Premature optimization is the root of all evil”*

:author: Gaël Varoquaux

This chapter deals with strategies to make Python code go faster.

.. topic:: Prerequisites

    * `line_profiler <http://packages.python.org/line_profiler/>`_
    * `gprof2dot <http://code.google.com/p/jrfonseca/wiki/Gprof2Dot>`_
    * `dot utility from Graphviz <http://www.graphviz.org/>`_

.. contents:: Chapters contents
   :local:
   :depth: 4


Optimization workflow
======================

#. Make it work: write the code in a simple **legible** ways.

#. Make it work reliably: write automated test cases, make really sure
   that your algorithm is right and that if you break it, the tests will
   capture the breakage.

#. Optimize the code by profiling simple use-cases to find the
   bottlenecks and speeding up these bottleneck, finding a better
   algorithm or implementation. Keep in mind that a trade off should be
   found between profiling on a realistic example and the simplicity and
   speed of execution of the code. For efficient work, it is best to work
   with profiling runs lasting around 10s.


Profiling Python code
=====================

.. topic:: **No optimization without measuring!**

    * **Measure:** profiling, timing

    * You'll have surprises: the fastest code is not always what you
      think


Timeit
---------

In IPython, use ``timeit`` (http://docs.python.org/library/timeit.html) to time elementary operations:

.. sourcecode:: ipython

    In [1]: import numpy as np

    In [2]: a = np.arange(1000)

    In [3]: %timeit a ** 2
    100000 loops, best of 3: 5.73 us per loop

    In [4]: %timeit a ** 2.1
    1000 loops, best of 3: 154 us per loop

    In [5]: %timeit a * a
    100000 loops, best of 3: 5.56 us per loop

Use this to guide your choice between strategies.

.. note::

   For long running calls, using ``%time`` instead of ``%timeit``; it is
   less precise but faster

Profiler
-----------

Useful when you have a large program to profile, for example the
:download:`following file <demo.py>`:

.. literalinclude:: demo.py


.. note::
    This is a combination of two unsupervised learning techniques, principal
    component analysis (`PCA
    <http://en.wikipedia.org/wiki/Principal_component_analysis>`_) and
    independent component analysis
    (`ICA <http://en.wikipedia.org/wiki/Independent_component_ana lysis>`_). PCA
    is a technique for dimensionality reduction, i.e. an algorithm to explain
    the observed variance in your data using less dimensions. ICA is a source
    seperation technique, for example to unmix multiple signals that have been
    recorded through multiple sensors. Doing a PCA first and then an ICA can be
    useful if you have more sensors than signals. For more information see:
    `the FastICA example from scikits-learn <http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html>`_.

To run it, you also need to download the :download:`ica module <ica.py>`.
In IPython we can time the script:

.. sourcecode:: ipython

   In [1]: %run -t demo.py

   IPython CPU timings (estimated):
       User  :    14.3929 s.
       System:   0.256016 s.

and profile it:

.. sourcecode:: ipython

   In [2]: %run -p demo.py

         916 function calls in 14.551 CPU seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   14.457   14.457   14.479   14.479 decomp.py:849(svd)
        1    0.054    0.054    0.054    0.054 {method 'random_sample' of 'mtrand.RandomState' objects}
        1    0.017    0.017    0.021    0.021 function_base.py:645(asarray_chkfinite)
       54    0.011    0.000    0.011    0.000 {numpy.core._dotblas.dot}
        2    0.005    0.002    0.005    0.002 {method 'any' of 'numpy.ndarray' objects}
        6    0.001    0.000    0.001    0.000 ica.py:195(gprime)
        6    0.001    0.000    0.001    0.000 ica.py:192(g)
       14    0.001    0.000    0.001    0.000 {numpy.linalg.lapack_lite.dsyevd}
       19    0.001    0.000    0.001    0.000 twodim_base.py:204(diag)
        1    0.001    0.001    0.008    0.008 ica.py:69(_ica_par)
        1    0.001    0.001   14.551   14.551 {execfile}
      107    0.000    0.000    0.001    0.000 defmatrix.py:239(__array_finalize__)
        7    0.000    0.000    0.004    0.001 ica.py:58(_sym_decorrelation)
        7    0.000    0.000    0.002    0.000 linalg.py:841(eigh)
      172    0.000    0.000    0.000    0.000 {isinstance}
        1    0.000    0.000   14.551   14.551 demo.py:1(<module>)
       29    0.000    0.000    0.000    0.000 numeric.py:180(asarray)
       35    0.000    0.000    0.000    0.000 defmatrix.py:193(__new__)
       35    0.000    0.000    0.001    0.000 defmatrix.py:43(asmatrix)
       21    0.000    0.000    0.001    0.000 defmatrix.py:287(__mul__)
       41    0.000    0.000    0.000    0.000 {numpy.core.multiarray.zeros}
       28    0.000    0.000    0.000    0.000 {method 'transpose' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.008    0.008 ica.py:97(fastica)
        ...

Clearly the ``svd`` (in `decomp.py`) is what takes most of our time, a.k.a. the
bottleneck. We have to find a way to make this step go faster, or to avoid this
step (algorithmic optimization). Spending time on the rest of the code is
useless.

Alternatively you can also use the ``%prun`` magic to profile a function call.
This is useful if you do not have an executable script but are profiling part
of a library. This magic has a useful switch ``-l`` which can be used to limit
the number of lines of output. (Shown below is a run from a different machine,
hence the difference in timing.)

.. sourcecode:: ipython

    In [44]: import demo

    In [45]: %prun -l 5 demo.test()

             286 function calls in 5.053 seconds

       Ordered by: internal time
       List reduced from 64 to 5 due to restriction <5>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    5.034    5.034    5.037    5.037 decomp_svd.py:15(svd)
            1    0.008    0.008    5.053    5.053 <string>:1(<module>)
            1    0.006    0.006    0.006    0.006 {method 'random_sample' of 'mtrand.RandomState' objects}
           14    0.002    0.000    0.002    0.000 {numpy.core._dotblas.dot}
            1    0.002    0.002    0.002    0.002 function_base.py:527(asarray_chkfinite)

Line-profiler
--------------

The profiler is great: it tells us which function takes most of the time,
but not where it is called.

For this, we use the
`line_profiler <http://packages.python.org/line_profiler/>`_: in the
source file, we decorate a few functions that we want to inspect with
``@profile`` (no need to import it)

.. sourcecode:: python

    @profile
    def test():
        data = np.random.random((5000, 100))
        u, s, v = linalg.svd(data)
        pca = np.dot(u[: , :10], data)
        results = fastica(pca.T, whiten=False)

Then we run the script using the `kernprof.py
<http://packages.python.org/line_profiler/kernprof.py>`_ program, with switches ``-l, --line-by-line`` and ``-v, --view`` to use the line-by-line profiler and view the results in addition to saving them:

.. sourcecode:: console

    $ kernprof.py -l -v demo.py

    Wrote profile results to demo.py.lprof
    Timer unit: 1e-06 s

    File: demo.py
    Function: test at line 5
    Total time: 14.2793 s

    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
        5                                           @profile
        6                                           def test():
        7         1        19015  19015.0      0.1      data = np.random.random((5000, 100))
        8         1     14242163 14242163.0   99.7      u, s, v = linalg.svd(data)
        9         1        10282  10282.0      0.1      pca = np.dot(u[:10, :], data)
       10         1         7799   7799.0      0.1      results = fastica(pca.T, whiten=False)

**The SVD is taking all the time.** We need to optimise this line.

Running ``cProfile``
--------------------

In the IPython example above, IPython simply calls the built-in `Python
profilers <http://docs.python.org/2/library/profile.html>`_ ``cProfile`` and
``profile``. This can be useful if you wish to process the profiler output with a
visualization tool.

.. sourcecode:: console

   $  python -m cProfile -o demo.prof demo.py

Using the ``-o`` switch will output the profiler results to the file
``demo.prof``.

Using ``gprof2dot``
-------------------

In case you want a more visual representation of the profiler output, you can
use the `gprof2dot <http://code.google.com/p/jrfonseca/wiki/Gprof2Dot>`_ tool:

.. sourcecode:: console

    $ gprof2dot -f pstats demo.prof | dot -Tpng -o demo-prof.png

Which will produce the following picture:

.. image:: demo-prof.png

Which again paints a similar picture as the previous approaches.

Making code go faster
======================

Once we have identified the bottlenecks, we need to make the
corresponding code go faster.

Algorithmic optimization
-------------------------

The first thing to look for is algorithmic optimization: are there ways
to compute less, or better?

For a high-level view of the problem, a good understanding of the maths
behind the algorithm helps. However, it is not uncommon to find simple
changes, like **moving computation or memory allocation outside a for
loop**, that bring in big gains.

Complexity Theory
.................

In order to asses the runtime complexity of algorithms a brief interlude is
needed. In computer science so called **Big-Oh** notation (`Wikipedia
<http://en.wikipedia.org/wiki/Big_O_notation>`_) is used to classify the
asymptotic complexity of algorithms in terms of the size of their input. While
a formal mathematical treatment is well beyond the scope of this tutorial, some
simple intuitive ideas about complexity can be enough to provide some insights.

A definition of Big-Oh is:

.. math::

   f(n) = \mathcal{O}(g(n))

means there are positive constants :math:`c` and :math:`k`, such that :math:`0
≤ f(n) ≤ cg(n)` for all :math:`n ≥ k`. The values of :math:`c` and :math:`k`
must be fixed for the function :math:`f` and must not depend on :math:`n`.

For example the implementation of max:

.. literalinclude:: max.py
   :lines: 6-11

As you can see, this algorithms needs to check each value once. Hence we say
that it is of linear complexity, or it is **Big-Oh of n**:

.. math::

   f(n) = \mathcal{O}(n)

Now, let's look at the following example of the insertion sort algorithm:

.. literalinclude:: insertion_sort.py
   :lines: 7-15

As you can see here we have two for loops. While the first loop iterates over
n, the second one iterates of the sequence :math:`(n-1)` the first time,
:math:`(n-2)` the second time, :math:`(n-3)` the third time and so on. This is
an arithmetic progression and the solution is:

.. math::

   \frac{n(n-1)}{2}

Since we are dealing with asymptotic complexity, we need only look at the
dominating term, i.e. the term that grows fastest in the expression, and thus
insertion sort has complexity:

.. math::

   f(n) = \mathcal{O}(n^2)

Note that the complexity analysis has been fairly easy in this case because the
number of comparisons does not depend on the order of the data. I.e. even if
the array is already sorted. Other sorting algorithms are not so easy in terms
of analysis and may have best-case, average-case and worst-case complexities.

Lastly, let's look at the binary search algorithm, that finds the index of an
element in a sorted array:

.. literalinclude:: binary_search.py
   :lines: 7-18

We can see that we keep partitioning the set into roughly half for every
iteration of the while loop. So we make at most  :math:`⌊log_{2}(n)+1⌋`
comparisons, leading to a complexity of:

.. math::

   f(n) = \mathcal{O}(log_{2}n)

Incidentally the best-case runtime is:

.. math::

   f(n) = \mathcal{O}(1)

And this happens when we find the desired element after the first comparison.


Example of the SVD
...................

In both examples above, the SVD -
`Singular Value Decomposition <http://en.wikipedia.org/wiki/Singular_value_decomposition>`_
- is what
takes most of the time. Indeed, the computational cost of this algorithm is
roughly :math:`n^3` in the size of the input matrix.

However, in both of these example, we are not using all the output of
the SVD, but only the first few rows of its first return argument. If
we use the ``svd`` implementation of scipy, we can ask for an incomplete
version of the SVD. Note that implementations of linear algebra in
scipy are richer then those in numpy and should be preferred.

.. sourcecode:: ipython

    In [3]: %timeit np.linalg.svd(data)
    1 loops, best of 3: 14.5 s per loop

    In [4]: from scipy import linalg

    In [5]: %timeit linalg.svd(data)
    1 loops, best of 3: 14.2 s per loop

    In [6]: %timeit linalg.svd(data, full_matrices=False)
    1 loops, best of 3: 295 ms per loop

    In [7]: %timeit np.linalg.svd(data, full_matrices=False)
    1 loops, best of 3: 293 ms per loop

We can then use this insight to :download:`optimize the previous code <demo_opt.py>`:

.. literalinclude:: demo_opt.py
   :start-line: 9
   :end-line: 13

.. sourcecode:: ipython

    In [1]: import demo

    In [2]: %timeit demo.
    demo.fastica   demo.np        demo.prof.pdf  demo.py        demo.pyc
    demo.linalg    demo.prof      demo.prof.png  demo.py.lprof  demo.test

    In [2]: %timeit demo.test()
    ica.py:65: RuntimeWarning: invalid value encountered in sqrt
      W = (u * np.diag(1.0/np.sqrt(s)) * u.T) * W  # W = (W * W.T) ^{-1/2} * W
    1 loops, best of 3: 17.5 s per loop

    In [3]: import demo_opt

    In [4]: %timeit demo_opt.test()
    1 loops, best of 3: 208 ms per loop

Real incomplete SVDs, e.g. computing only the first 10 eigenvectors, can
be computed with arpack, available in ``scipy.sparse.linalg.eigsh``.

.. topic:: Computational linear algebra

    For certain algorithms, many of the bottlenecks will be linear
    algebra computations. In this case, using the right function to solve
    the right problem is key. For instance, an eigenvalue problem with a
    symmetric matrix is easier to solve than with a general matrix. Also,
    most often, you can avoid inverting a matrix and use a less costly
    (and more numerically stable) operation.

    Know your computational linear algebra. When in doubt, explore
    ``scipy.linalg``, and use ``%timeit`` to try out different alternatives
    on your data.

Writing faster numerical code
===============================

A complete discussion on advanced use of numpy is found in chapter
:ref:`advanced_numpy`, or in the article `The NumPy array: a structure
for efficient numerical computation
<http://hal.inria.fr/inria-00564007/en>`_ by van der Walt et al. Here we
discuss only some commonly encountered tricks to make code faster.

* **Vectorizing for loops**

  Find tricks to avoid for loops using numpy arrays. For this, masks and
  indices arrays can be useful.

* **Broadcasting**

  Use :ref:`broadcasting <broadcasting>` to do operations on arrays as
  small as possible before combining them.

.. XXX: complement broadcasting in the numpy chapter with the example of
   the 3D grid

* **In place operations**

  .. sourcecode:: ipython

    In [1]: a = np.zeros(1e7)

    In [2]: %timeit global a ; a = 0*a
    10 loops, best of 3: 111 ms per loop

    In [3]: %timeit global a ; a *= 0
    10 loops, best of 3: 48.4 ms per loop

  **note**: we need `global a` in the timeit so that it work, as it is
  assigning to `a`, and thus considers it as a local variable.

* **Be easy on the memory: use views, and not copies**

  Copying big arrays is as costly as making simple numerical operations
  on them:

  .. sourcecode:: ipython

    In [1]: a = np.zeros(1e7)

    In [2]: %timeit a.copy()
    10 loops, best of 3: 124 ms per loop

    In [3]: %timeit a + 1
    10 loops, best of 3: 112 ms per loop

* **Beware of cache effects**

  Memory access is cheaper when it is grouped: accessing a big array in a
  continuous way is much faster than random access. This implies amongst
  other things that **smaller strides are faster** (see
  :ref:`cache_effects`):

  .. sourcecode:: ipython

    In [1]: c = np.zeros((1e4, 1e4), order='C')

    In [2]: %timeit c.sum(axis=0)
    1 loops, best of 3: 3.89 s per loop

    In [3]: %timeit c.sum(axis=1)
    1 loops, best of 3: 188 ms per loop

    In [4]: c.strides
    Out[4]: (80000, 8)

  This is the reason why Fortran ordering or C ordering may make a big
  difference on operations:

  .. sourcecode:: ipython

    In [5]: a = np.random.rand(20, 2**18)

    In [6]: b = np.random.rand(20, 2**18)

    In [7]: %timeit np.dot(b, a.T)
    1 loops, best of 3: 194 ms per loop

    In [8]: c = np.ascontiguousarray(a.T)

    In [9]: %timeit np.dot(b, c)
    10 loops, best of 3: 84.2 ms per loop

  Note that copying the data to work around this effect may not be worth it:

  .. sourcecode:: ipython

    In [10]: %timeit c = np.ascontiguousarray(a.T)
    10 loops, best of 3: 106 ms per loop

  Using `numexpr <http://code.google.com/p/numexpr/>`_ can be useful to
  automatically optimize code for such effects.

* **Use compiled code**

  The last resort, once you are sure that all the high-level
  optimizations have been explored, is to transfer the hot spots, i.e.
  the few lines or functions in which most of the time is spent, to
  compiled code. For compiled code, the preferred option is to use
  `Cython <http://www.cython.org>`_: it is easy to transform exiting
  Python code in compiled code, and with a good use of the
  `numpy support <http://docs.cython.org/src/tutorial/numpy.html>`_
  yields efficient code on numpy arrays, for instance by unrolling loops.

.. warning::

   For all the above: profile and time your choices. Don't base your
   optimization on theoretical considerations.

Hardware... Software...
-----------------------

Lastly, let's look at an interesting optimization in Numpy. The optimization is
that the power operator ``**`` checks if the exponent is two (square) and then
does a multiplication avoiding the expensive ``power`` implementation which
presumably takes care of all sorts of edge cases such as floating point
exponents etc.. We say that multiplication is *implemented in hardware* whereas
power is *implemented in software*.

.. sourcecode:: ipython

    In [4]: %timeit np.power(a, 2)
    100 loops, best of 3: 17.8 ms per loop

    In [5]: %timeit a ** 2
    1000 loops, best of 3: 1.55 ms per loop

    In [6]: %timeit a * a
    1000 loops, best of 3: 1.53 ms per loop

As a comparison, here are the timings for the cube:

.. sourcecode:: ipython

    In [7]: %timeit np.power(a, 3)
    10 loops, best of 3: 78.7 ms per loop

    In [8]: %timeit a ** 3
    10 loops, best of 3: 78.7 ms per loop

    In [9]: %timeit a * a * a
    100 loops, best of 3: 6.02 ms per loop

As you can see, implementing this with multiplication is much faster, but the
Numpy code base is not optimized for this case.

Lastly, to instill some curiosity let's look at the timings we get when using
the `numexpr <http://code.google.com/p/numexpr/>`_ tool:

.. sourcecode:: ipython

    In [21]: %timeit numexpr.evaluate('a ** 3')
    1000 loops, best of 3: 1.7 ms per loop

As you can see even the cube is much faster with numexpr than with any of the
three previous approaches. Can you figure out why? How does it look for square?


Additional Links
----------------

* If you need to profile memory usage, you could try the `memory_profiler
  <http://pypi.python.org/pypi/memory_profiler>`_

* If you need to profile down into C extensions, you could try using
  `gperftools <http://code.google.com/p/gperftools/?redir=1>`_ from Python with
  `yep <http://pypi.python.org/pypi/yep>`_.

* If you would like to track performance of your code across time, i.e. as you
  make new commits to your repository, you could try:
  `vbench <https://github.com/pydata/vbench>`_

* If you need some interactive visualization why not try `RunSnakeRun
  <http://www.vrplumber.com/programming/runsnakerun/>`_

* `Article by Wes McKinney on how he made Pandas so fast <http://wesmckinney.com/blog/?p=489>`_

* `Big-Oh complexity of various python operations <https://wiki.python.org/moin/TimeComplexity>`_
