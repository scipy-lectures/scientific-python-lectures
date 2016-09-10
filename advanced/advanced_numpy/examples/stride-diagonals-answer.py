"""
Solution to the stride diagonal exercise
=========================================

Solution showing how to use as_strided to stride in diagonal.

"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

#
# Part 1
#

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.int32)

x_diag = as_strided(x, shape=(3,), strides=((3+1)*x.itemsize,))
x_supdiag = as_strided(x[0,1:], shape=(2,), strides=((3+1)*x.itemsize,))
x_subdiag = as_strided(x[1:,0], shape=(2,), strides=((3+1)*x.itemsize,))

print(x_diag)
print(x_supdiag)
print(x_subdiag)

#
# Mini-exercise: (assume C memory order)
#
# 0. How to pick diagonal entries of the matrix
#
# 1. How to pick the super-diagonal entries [2, 6]
#
# 2. The sub-diagonal entries [4, 8]
#
# 99. Can you generalize this for any stride and shape combinations
#     in the initial array?
#
#     If you can, tell me, and maybe numpy.trace can be made faster :)
#


#
# Part 2
#

# Compute the tensor trace

x = np.arange(5*5*5*5).reshape(5,5,5,5)

s = 0
for i in range(5):
    for j in range(5):
        s += x[j,i,j,i]

# by striding and using .sum()

y = as_strided(x, shape=(5, 5), strides=((5*5*5+5)*x.itemsize,
                                         (5*5+1)*x.itemsize))
s2 = y.sum()

assert s == s2
