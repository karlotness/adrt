from setuptools import setup, Extension
import numpy
import glob
import sys


adrt_c_ext = Extension(
    "adrt._adrt_cdefs",
    sources=glob.glob("src/adrt/*.cpp"),
    depends=glob.glob("src/adrt/*.hpp"),
    language="c++",
    include_dirs=[numpy.get_include()],
    py_limited_api=True,
    define_macros=[
        ("Py_LIMITED_API", "0x03080000"),
        ("NPY_NO_DEPRECATED_API", "NPY_1_17_API_VERSION"),
    ],
)


setup(
    ext_modules=[adrt_c_ext],
    options={
        "bdist_wheel": {
            # Automatically build wheels for current Python version and later.
            # Override by providing command-line arguments to bdist_wheel.
            # Combined with oldest-supported-numpy this ensures that
            # built wheels use an old NumPy version.
            "py_limited_api": f"cp{sys.version_info.major}{sys.version_info.minor}",
        },
    },
)
