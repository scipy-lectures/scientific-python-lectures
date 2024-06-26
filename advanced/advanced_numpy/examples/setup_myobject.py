"""
Building the extension
=======================

The script to build the extension

"""

from setuptools import setup, Extension

setup(
    name="myobject",
    version="1.0",
    ext_modules=[Extension("myobject", ["myobject.c"])],
)
