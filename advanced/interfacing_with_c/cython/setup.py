from distutils.core import Extension, setup

from Cython.Distutils import build_ext

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[Extension("cos_module", ["cos_module.pyx"])],
)
