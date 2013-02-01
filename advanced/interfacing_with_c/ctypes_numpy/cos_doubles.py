import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int

array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

libcd = npct.load_library("libcos_doubles", ".")
libcd.cos_doubles.restype = None
libcd.cos_doubles.argtypes = [array_1d_double, array_1d_double, c_int]

cos_doubles_func = libcd.cos_doubles
