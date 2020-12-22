from setuptools import setup, Extension
import numpy
import glob


adrt_c_ext = Extension(
    "adrt._adrt_cdefs",
    sources=glob.glob("src/adrt/*.cpp"),
    depends=glob.glob("src/adrt/*.hpp"),
    language="c++",
    include_dirs=[numpy.get_include()],
    py_limited_api=True,
    define_macros=[
        ("Py_LIMITED_API", "0x03060000"),
        ("NPY_NO_DEPRECATED_API", "NPY_1_17_API_VERSION"),
    ],
)


setup(
    ext_modules=[adrt_c_ext],
)
