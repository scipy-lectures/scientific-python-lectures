"""
Building the extension
=======================

The script to build the extension

"""

from distutils.core import Extension, setup

setup(
    name="myobject",
    version="1.0",
    ext_modules=[Extension("myobject", ["myobject.c"])],
)
