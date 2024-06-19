from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "cos_module",
        sources=["cos_module.pyx"]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
