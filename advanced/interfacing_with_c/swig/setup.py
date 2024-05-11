from distutils.core import Extension, setup

setup(ext_modules=[Extension("_cos_module", sources=["cos_module.c", "cos_module.i"])])
