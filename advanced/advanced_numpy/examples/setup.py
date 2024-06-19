"""
Building the mandel C-Python extension
=======================================

The "setup.py" script that builds the mandel.so extension from the
C sources.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "mandel",
        sources=["mandel.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
