import numpy as np
from numpy.lib.stride_tricks import as_strided

x = np.array([1, 2, 3, 4], dtype=np.int8)

#
# Mini-exercise:
#
# 1. How to create a new array that shares the data, but looks like
#
#    array([[1, 2, 3, 4],
#           [1, 2, 3, 4],
#           [1, 2, 3, 4]], dtype=int8)
#
