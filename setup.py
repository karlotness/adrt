
from distutils.core import setup, Extension
import numpy, os

numpy_dir = os.path.join(numpy.get_include(),'numpy')
ext1 = Extension('adrt._adrtc',
                 include_dirs=[numpy_dir],
                 sources = ['adrt/adrt.c'])

setup(name = 'adrtc',
      packages = ['adrt'],
      version = '0.1.0',
      description       = "Approximate Discrete Radon Transform",
      author            = "ADRT Development Team",
      ext_modules = [ext1]
      )
