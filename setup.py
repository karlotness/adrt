from setuptools import setup, find_packages, Extension
import numpy
import glob

adrt_c_ext = Extension("adrt._adrt_cdefs",
                       sources=glob.glob('adrt/*.cpp'),
                       depends=glob.glob('adrt/*.hpp'),
                       language='c++',
                       include_dirs=[numpy.get_include()],
                       py_limited_api=True,
                       define_macros=[
                           ("Py_LIMITED_API", "0x03050000"),
                           ("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")])

setup(name="adrt",
      description="Fast approximate discrete Radon transform for NumPy arrays",
      version="0.1.0",
      packages=find_packages(),
      python_requires=">=3.5, <4",
      install_requires=["numpy>=1.9"],
      license="BSD",
      ext_modules=[adrt_c_ext])
