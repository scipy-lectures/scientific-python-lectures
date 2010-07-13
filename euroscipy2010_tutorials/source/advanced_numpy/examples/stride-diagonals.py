import numpy as np
from numpy.lib.stride_tricks import as_strided

#
# Part 1
#

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.int32)

x_diag = as_strided(x, shape=(3,), strides=(TODO,))
x_supdiag = TODO
x_subdiag = TODO

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
for i in xrange(5):
    for j in xrange(5):
        s += x[j,i,j,i]

# by striding and using .sum()

y = as_strided(x, shape=(5, 5), strides=(TODO, TODO))
s2 = ...

assert s == s2
