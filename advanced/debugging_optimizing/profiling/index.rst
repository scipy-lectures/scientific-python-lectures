Profiling Python code
==========================================

.. topic:: **No optimization without measuring!**

    * **Measure:** profiling, timing

    * "Premature optimization is the root of all evil"


__________

Timeit
---------

In IPython, to time elementatry operations:

.. sourcecode:: ipython
    
    In [1]: import numpy as np

    In [2]: a = np.arange(1000)

    In [3]: %timeit a**2
    100000 loops, best of 3: 5.73 us per loop

    In [4]: %timeit a**2.1
    1000 loops, best of 3: 154 us per loop

    In [5]: %timeit a*a
    100000 loops, best of 3: 5.56 us per loop

Profiler
-----------

Useful when you have a large program to profile.

.. literalinclude:: demo.py

.. sourcecode:: ipython

    In [1]: %run -t demo.py

    IPython CPU timings (estimated):
	User  :    14.3929 s.
        System:   0.256016 s.

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
       25    0.000    0.000    0.000    0.000 {abs}
       19    0.000    0.000    0.000    0.000 {numpy.core.multiarray.arange}
       21    0.000    0.000    0.000    0.000 defmatrix.py:527(getT)
        7    0.000    0.000    0.000    0.000 linalg.py:64(_commonType)
       41    0.000    0.000    0.000    0.000 {len}
       13    0.000    0.000    0.000    0.000 {max}
        9    0.000    0.000    0.000    0.000 {method 'view' of 'numpy.ndarray' objects}
       28    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
       14    0.000    0.000    0.000    0.000 linalg.py:36(isComplexType)
       23    0.000    0.000    0.000    0.000 {issubclass}
        7    0.000    0.000    0.000    0.000 linalg.py:92(_fastCopyAndTranspose)
       14    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
       14    0.000    0.000    0.000    0.000 linalg.py:49(_realType)
        7    0.000    0.000    0.000    0.000 {numpy.core.multiarray._fastCopyAndTranspose}
        7    0.000    0.000    0.000    0.000 linalg.py:31(_makearray)
        7    0.000    0.000    0.000    0.000 linalg.py:110(_assertSquareness)
        1    0.000    0.000    0.000    0.000 lapack.py:63(get_lapack_funcs)
        1    0.000    0.000    0.000    0.000 lapack.py:48(find_best_lapack_type)
        7    0.000    0.000    0.000    0.000 linalg.py:104(_assertRank2)
       15    0.000    0.000    0.000    0.000 {min}
        7    0.000    0.000    0.000    0.000 {method '__array_wrap__' of 'numpy.ndarray' objects}
        6    0.000    0.000    0.000    0.000 defmatrix.py:521(getA)


Line-profiler
--------------

::

    @profile
    def test():
	data = np.random.random((5000, 100))
	u, s, v = linalg.svd(data)
	pca = np.dot(u[:10, :], data) 
	results = fastica(pca.T, whiten=False)

::

    ~ $ kernprof.py -l -v demo.py

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


**The SVD is taking all the time.** We need to optimise this ligne.


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

