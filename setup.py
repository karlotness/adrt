from setuptools import setup, find_packages, Extension
import numpy

adrtc_ext = Extension("adrt._adrtc",
                      sources=["adrt/adrt.c"],
                      include_dirs=[numpy.get_include()])

setup(name="adrtc",
      description="Approximate Discrete Radon Transform",
      version="0.1.0",
      packages=find_packages(),
      python_requires=">=3.4",
      install_requires=["numpy>=1.8"],
      url="https://github.com/dsrim/adrtc",
      license="BSD",
      ext_modules=[adrtc_ext])
