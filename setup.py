#encoding:utf-8
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("evalueteC", ["*.pyx"],
        include_dirs = [numpy.get_include()])
]

setup(
    name = "eva",
    ext_modules = cythonize(extensions)
)