"""Example of wrapping cos function from math.h using ctypes."""

import ctypes

# find and load the library

# OSX or linux
from ctypes.util import find_library

libm_name = find_library("m")
assert libm_name is not None, "Cannot find libm (math) on this system :/ That's bad."

libm = ctypes.cdll.LoadLibrary(libm_name)

# Windows
# from ctypes import windll
# libm = cdll.msvcrt


# set the argument type
libm.cos.argtypes = [ctypes.c_double]
# set the return type
libm.cos.restype = ctypes.c_double


def cos_func(arg):
    """Wrapper for cos from math.h"""
    return libm.cos(arg)
