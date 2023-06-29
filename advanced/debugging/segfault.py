""" Simple code that creates a segfault using numpy. Used to learn
debugging segfaults with GDB.
"""

import numpy as np
from numpy.lib import stride_tricks


def make_big_array(small_array):
    big_array = stride_tricks.as_strided(
        small_array, shape=(2e6, 2e6), strides=(32, 32)
    )
    return big_array


def print_big_array(small_array):
    big_array = make_big_array(small_array)
    print(big_array[-10:])
    return big_array


l = []
for i in range(10):
    a = np.arange(8)
    l.append(print_big_array(a))
