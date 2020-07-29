from setuptools import setup, find_packages, Extension
import numpy
import glob

adrtc_ext = Extension("adrtc._adrtc_cdefs",
                      sources=["adrtc/adrtc_cdefs.cpp"],
                      depends=glob.glob('adrtc/*.hpp'),
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp'],
                      include_dirs=[numpy.get_include()])

setup(name="adrtc",
      description="Approximate Discrete Radon Transform",
      version="0.1.0",
      packages=find_packages(),
      python_requires=">=3.4, <4",
      install_requires=["numpy>=1.8"],
      license="BSD",
      ext_modules=[adrtc_ext])
