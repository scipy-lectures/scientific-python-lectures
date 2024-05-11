from distutils.core import Extension, setup

from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "cos_doubles",
            sources=["_cos_doubles.pyx", "cos_doubles.c"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
