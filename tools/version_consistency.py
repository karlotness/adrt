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


import argparse
import re
import ast
import sys
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
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


# This maps Python version strings to the oldest version of NumPy
# (without patch version) on PyPI to have binary wheels for that
# release
PYTHON_TO_MIN_NUMPY_MAP = {
    "3.9": "1.19",
    "3.10": "1.21",
    "3.11": "1.23",
    "3.12": "1.26",
    "3.13": "2.1",
}


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
    if re_match := macro_re.search(content):
        # Normalize tokens
        return " ".join(re_match.group("val").strip().split())
    raise ValueError(f"Could not find macro {macro}")


def find_global_variable_def(var, py_path):
    with open(py_path, "r", encoding="utf8") as py_file:
        content = py_file.read()
    assignments = [
        node
        for node in ast.walk(ast.parse(content))
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == var for t in node.targets)
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == var
        )
    ]
    if len(assignments) != 1:
        raise ValueError(
            f"Could not locate unique {var} assignment, found {len(assignments)}"
        )
    ver_assign = assignments.pop()
    if ver_assign.value is None:
        raise ValueError(f"{var} assignment missing value")
    return ast.literal_eval(ver_assign.value)


def find_package_version(init_py):
    ver_str = find_global_variable_def("__version__", init_py)
    if not isinstance(ver_str, str):
        raise ValueError(f"Version attribute is not a string {ver_str}")
    return ver_str


def find_release_tag_version(tag_string):
    if tag_string is None:
        return None
    ver_re = re.compile(r"refs/tags/v(?P<ver>\S+)")
    if re_match := ver_re.fullmatch(tag_string):
        return re_match.group("ver")
    else:
        raise ValueError(f"Invalid tag format {tag_string}")


def find_meta_min_python(pyproject_toml):
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomllib.load(pyproj_file)
    ver_constraint = defs["project"]["requires-python"]
    return find_min_version("python", ["python" + ver_constraint])


def find_limited_api_python(setup_py):
    ver_str = find_global_variable_def("LIMITED_API_VERSION", setup_py)
    if not isinstance(ver_str, str):
        raise ValueError(f"Limited API value is not a string {ver_str}")
    if not re.fullmatch(r"\d+\.\d+", ver_str, re.ASCII):
        raise ValueError(f"Limited API value is not formatted correctly {ver_str}")
    return ver_str


def find_package_min_numpy(pyproject_toml):
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomllib.load(pyproj_file)
    ver_constraint = defs["project"]["dependencies"]
    return find_min_version("numpy", filter(bool, map(str, ver_constraint)))


def numpy_version_from_macro(py_cpp, macro):
    macro_text = find_cpp_macro_def(macro, py_cpp)
    rgx = re.compile(r"NPY_(?P<major>\d+)_(?P<minor>\d+)_API_VERSION")
    if re_match := rgx.fullmatch(macro_text):
        major = int(re_match.group("major"))
        minor = int(re_match.group("minor"))
        # NumPy<1.18 is missing some API version definitions.
        # There were no changes in this range so we can safely round up
        if (major, minor) <= (1, 17):
            major = 1
            minor = 17
        return canonicalize_version(f"{major}.{minor}")
    else:
        raise ValueError(f"Invalid NumPy API macro: {macro_text}")


def find_build_min_numpy(pyproject_toml):
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomllib.load(pyproj_file)
    ver_constraint = defs["build-system"]["requires"]
    return find_min_version("numpy", filter(bool, map(str, ver_constraint)))


def find_inconsistent_python_classifiers(pyproject_toml):
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomllib.load(pyproj_file)
    rgx = re.compile(r"Programming Language :: Python :: (?P<version>\d+\.\d+)")
    min_python = SpecifierSet(defs["project"]["requires-python"])
    inconsistent_versions = set()
    for classifier in defs["project"]["classifiers"]:
        if (re_match := rgx.fullmatch(classifier)) and Version(
            re_match.group("version")
        ) not in min_python:
            inconsistent_versions.add(re_match.group("version"))
    return inconsistent_versions


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
    # Check package classifiers
    if wrong_classifiers := find_inconsistent_python_classifiers("pyproject.toml"):
        class_str = ", ".join(wrong_classifiers)
        print(f"Python classifiers do not match requires-python (remove {class_str})")
        failure = True
    print("")

    # Check Python version requirements
    meta_min_python = find_meta_min_python("pyproject.toml")
    limited_api_python = find_limited_api_python("setup.py")
    print(f"Metadata min Python: {meta_min_python}")
    print(f"Limited API Python: {limited_api_python}")
    # Check consistency (Minimum Python versions should all be the same)
    if meta_min_python != limited_api_python:
        print("Python version mismatch")
        failure = True
    print("")

    # Check NumPy version requirements
    package_min_numpy = find_package_min_numpy("pyproject.toml")
    macro_numpy_target = numpy_version_from_macro(
        "src/adrt/adrt_cdefs_py.cpp", "NPY_TARGET_VERSION"
    )
    build_min_numpy = find_build_min_numpy("pyproject.toml")
    macro_numpy_deprecated = numpy_version_from_macro(
        "src/adrt/adrt_cdefs_py.cpp", "NPY_NO_DEPRECATED_API"
    )
    print(f"Package min NumPy: {package_min_numpy}")
    print(f"Macro NumPy API Target: {macro_numpy_target}")
    print(f"Build min NumPy: {build_min_numpy}")
    print(f"Macro NumPy Deprecated API: {macro_numpy_deprecated}")
    # The target API version should match the earliest NumPy release
    # supporting our minimum version of Python (this makes explicit
    # NumPy's default behavior) and ensures that all wheels we build
    # are in fact compatible with all supported versions of Python and
    # NumPy regardless of which version of Python was used to do the
    # actual build. The macro must be set equal to this value to keep
    # Conda-Forge builds simple. Don't use a newer version for the
    # macro.
    if macro_numpy_target != canonicalize_version(
        PYTHON_TO_MIN_NUMPY_MAP[meta_min_python]
    ):
        print(
            "NumPy target runtime API does not support oldest possible version"
            f" (should use {PYTHON_TO_MIN_NUMPY_MAP[meta_min_python]})"
        )
        failure = True
    # The version used to build must not be older than the target API version
    if Version(build_min_numpy) < Version(macro_numpy_target):
        print("Minimum build version must be at least the API target version")
        failure = True
    # The version used at runtime must not be older than the target API version
    if Version(package_min_numpy) < Version(macro_numpy_target):
        print("Runtime version should be at least the target API version")
        failure = True
    # We want to warn on API deprecated since our minimum build version
    if build_min_numpy != macro_numpy_deprecated:
        print("NumPy build and deprecation API version mismatch")
        failure = True
    # We need to use at least NumPy 1.25 to build (for API version support)
    if Version(build_min_numpy) < Version("1.25"):
        print("NumPy >=1.25 required for API target version")
        failure = True
    # We should always be able to use a build version that is at least
    # the runtime version (otherwise we can use a newer NumPy to build)
    if Version(package_min_numpy) > Version(build_min_numpy):
        print("Runtime NumPy version is newer than the build version")
        failure = True

    # Make sure versions match
    sys.exit(1 if failure else 0)
