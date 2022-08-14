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


import argparse
import re
import ast
import sys
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from packaging.utils import canonicalize_name, canonicalize_version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


parser = argparse.ArgumentParser(description="Check version strings for consistency")
parser.add_argument(
    "--tag_ref",
    type=str,
    default=None,
    help="The tag reference being deployed (if any)",
)
parser.add_argument(
    "--not_dev",
    action="store_true",
    help="Fail on development versions",
)


def is_valid_version(ver_str):
    try:
        _ = Version(ver_str)
    except InvalidVersion:
        return False
    return True


def is_dev_version(ver_str):
    try:
        return Version(ver_str).is_devrelease
    except InvalidVersion:
        return False


def find_min_version(package, requirements):
    min_operators = {">=", "~=", "=="}
    found_versions = []
    for req_str in requirements:
        req = Requirement(req_str)
        if canonicalize_name(req.name) == package:
            # This is the right package
            for spec in req.specifier:
                if spec.operator in min_operators:
                    ver = Version(spec.version)
                    found_versions.append(ver)
    if not found_versions:
        raise ValueError(f"Could not find minimum version for {package}")
    return str(canonicalize_version(min(found_versions)))


def find_cpp_macro_def(macro, cpp_path):
    macro_re = re.compile(
        rf"^\s*#\s*define\s+{re.escape(macro)}\s(?P<val>.+)$", re.MULTILINE
    )
    with open(cpp_path, "r", encoding="utf8") as cpp_file:
        content = cpp_file.read()
    if match := macro_re.search(content):
        # Normalize tokens
        return " ".join(match.group("val").strip().split())
    raise ValueError(f"Could not find macro {macro}")


def find_package_version(init_py):
    with open(init_py, "r", encoding="utf8") as init_file:
        content = init_file.read()
    version_re = re.compile(r"^__version__\s*=(?P<ver>.+)$", re.MULTILINE)
    if match := version_re.search(content):
        ver_str = ast.literal_eval(match.group("ver"))
        if not isinstance(ver_str, str):
            raise ValueError(f"Version attribute is not a string {ver_str}")
        return ver_str
    raise ValueError("Could not find package __version__ attribute")


def find_release_tag_version(tag_string):
    if tag_string is None:
        return None
    ver_re = re.compile(r"^refs/tags/v(?P<ver>.+)$")
    if match := ver_re.match(tag_string):
        return match.group("ver")
    else:
        raise ValueError(f"Invalid tag format {tag_string}")


def find_meta_min_python(pyproject_toml):
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomllib.load(pyproj_file)
    ver_constraint = defs["project"]["requires-python"]
    return find_min_version("python", ["python" + ver_constraint])


def find_macro_min_python(py_cpp):
    min_python = find_cpp_macro_def("Py_LIMITED_API", py_cpp)
    if not min_python.startswith("0x") or len(min_python) != 10:
        raise ValueError(f"Limited API macro is not a valid hex string: {min_python}")
    min_python = int(min_python, base=16)
    major = (min_python >> 24) & 0xFF
    minor = (min_python >> 16) & 0xFF
    micro = (min_python >> 8) & 0xFF
    release = min_python & 0xFF
    if release != 0:
        raise ValueError("Using pre-release version for limited API")
    ver = f"{major}.{minor}.{micro}"
    return str(canonicalize_version(ver))


def find_package_min_numpy(pyproject_toml):
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomllib.load(pyproj_file)
    ver_constraint = defs["project"]["dependencies"]
    return find_min_version("numpy", filter(bool, map(str, ver_constraint)))


def find_macro_numpy_api(py_cpp):
    min_numpy = find_cpp_macro_def("NPY_NO_DEPRECATED_API", py_cpp)
    rgx = re.compile(r"^NPY_(?P<major>\d+)_(?P<minor>\d+)_API_VERSION$")
    if match := rgx.match(min_numpy):
        major = int(match.group("major"))
        minor = int(match.group("minor"))
        # NumPy<1.18 is missing some API version definitions.
        # There were no changes in this range so we can safely round up
        if (major, minor) <= (1, 17):
            major = 1
            minor = 17
        return f"{major}.{minor}"
    else:
        raise ValueError(f"Invalid NumPy API macro: {min_numpy}")


if __name__ == "__main__":
    args = parser.parse_args()
    failure = False
    # Check declared package version
    var_version = find_package_version("src/adrt/__init__.py")
    tag_version = find_release_tag_version(args.tag_ref)
    print(f"Package variable version: {var_version}")
    if tag_version is not None:
        print(f"Release tag version: {tag_version}")
    # Check format
    if var_version.strip() != var_version:
        print("Package version string has extra spaces")
        failure = True
    if var_version.lower() != var_version:
        print("Package version is not lower case")
        failure = True
    # Check consistency
    if not is_valid_version(var_version):
        print("Package version is invalid (see PEP 440)")
        failure = True
    if tag_version is not None and tag_version != var_version:
        print("Package version mismatch")
        failure = True
    if args.not_dev and is_dev_version(var_version):
        print("Package version is a development release")
        failure = True
    print("")

    # Check Python version requirements
    meta_min_python = find_meta_min_python("pyproject.toml")
    macro_limited_api = find_macro_min_python("src/adrt/adrt_cdefs_py.cpp")
    print(f"Metadata min Python: {meta_min_python}")
    print(f"Limited API macro: {macro_limited_api}")
    # Check consistency
    if meta_min_python != macro_limited_api:
        print("Python version mismatch")
        failure = True
    print("")

    # Check NumPy version requirements
    package_min_numpy = find_package_min_numpy("pyproject.toml")
    macro_min_numpy = find_macro_numpy_api("src/adrt/adrt_cdefs_py.cpp")
    print(f"Package min NumPy: {package_min_numpy}")
    print(f"Macro min NumPy: {macro_min_numpy}")
    if package_min_numpy != macro_min_numpy:
        print("NumPy version mismatch")
        failure = True

    # Make sure versions match
    sys.exit(1 if failure else 0)
