from setuptools import setup, find_packages, Extension
import numpy
import glob

adrt_c_ext = Extension("adrt._adrt_cdefs",
                       sources=["adrt/adrt_cdefs.cpp"],
                       depends=glob.glob('adrt/*.hpp'),
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'],
                       include_dirs=[numpy.get_include()],
                       py_limited_api=True,
                       define_macros=[("Py_LIMITED_API", "0x03040000")])

setup(name="adrt",
      description="Fast approximate discrete Radon transform for NumPy arrays",
      version="0.1.0",
      packages=find_packages(),
      python_requires=">=3.4, <4",
      install_requires=["numpy>=1.8"],
      license="BSD",
      ext_modules=[adrt_c_ext])
