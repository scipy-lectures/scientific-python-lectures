from distutils.core import Extension, setup

# define the extension module
cos_module = Extension("cos_module", sources=["cos_module.c"])

# run the setup
setup(ext_modules=[cos_module])
