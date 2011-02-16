from distutils.core import setup, Extension
setup(name='myobject',
      version='1.0',
      ext_modules=[Extension('myobject', ['myobject.c'])],
)
