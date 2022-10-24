# Copyright (c) 2022 Karl Otness, Donsub Rim
# All rights reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import glob
import sys


COMPILER_EXTRA_ARGS = {
    "unix": ["-std=c++11"],
    "msvc": ["/std:c++14"],
}


class CPPVersionBuildExt(build_ext):
    def build_extension(self, ext, *args, **kwargs):
        if ext.language == "c++":
            extra_args = COMPILER_EXTRA_ARGS.get(self.compiler.compiler_type, [])
            ext.extra_compile_args = extra_args + ext.extra_compile_args
        return super().build_extension(ext, *args, **kwargs)


adrt_c_ext = Extension(
    "adrt._adrt_cdefs",
    sources=glob.glob("src/adrt/*.cpp"),
    depends=glob.glob("src/adrt/*.hpp"),
    language="c++",
    include_dirs=[numpy.get_include()],
    py_limited_api=True,
)


setup(
    ext_modules=[adrt_c_ext],
    cmdclass={"build_ext": CPPVersionBuildExt},
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
