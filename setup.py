# Copyright 2023 Karl Otness, Donsub Rim
#
# SPDX-License-Identifier: BSD-3-Clause
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
import sysconfig


LIMITED_API_VERSION = "3.10"
FREE_THREADING_BUILD = sysconfig.get_config_var("Py_GIL_DISABLED")
COMPILER_EXTRA_ARGS = {
    "unix": ["-std=c++20"],
    "msvc": ["/std:c++20"],
}


class CPPVersionBuildExt(build_ext):
    def build_extension(self, ext, *args, **kwargs):
        if ext.language == "c++":
            extra_args = COMPILER_EXTRA_ARGS.get(self.compiler.compiler_type, [])
            ext.extra_compile_args = extra_args + ext.extra_compile_args
        return super().build_extension(ext, *args, **kwargs)


def build_extension_def():
    macro_defs = []
    if not FREE_THREADING_BUILD:
        major, minor = map(int, LIMITED_API_VERSION.split("."))
        macro_defs.append(("Py_LIMITED_API", f"0x{major:02X}{minor:02X}0000"))
    return Extension(
        "adrt._adrt_cdefs",
        sources=glob.glob("src/adrt/*.cpp"),
        depends=glob.glob("src/adrt/*.hpp"),
        language="c++",
        include_dirs=[numpy.get_include()],
        py_limited_api=(not FREE_THREADING_BUILD),
        define_macros=macro_defs,
    )


def build_default_options():
    options = {}
    if not FREE_THREADING_BUILD:
        major, minor = map(int, LIMITED_API_VERSION.split("."))
        options["bdist_wheel"] = {"py_limited_api": f"cp{major}{minor}"}
    return options


setup(
    ext_modules=[build_extension_def()],
    cmdclass={"build_ext": CPPVersionBuildExt},
    options=build_default_options(),
)
