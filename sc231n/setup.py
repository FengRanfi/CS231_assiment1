from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "im2col_cython", ["im2col_cython.pyx"], include_dirs=[numpy.get_include()]
    ),
]

setup(
    ext_modules = cythonize("im2col_cython.pyx")
)

setup(ext_modules=cythonize(extensions),
    include_dirs = [numpy.get_include(), "<path_to_python_include>"],
    library_dirs = ["<path_to_python_lib>"]
)
