from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        "cos_doubles",
        sources=["_cos_doubles.pyx", "cos_doubles.c"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
