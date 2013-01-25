cdef extern from "math.h":
    double cos(double arg)

def cos_func(arg):
    return cos(arg)
