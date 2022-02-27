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
import tomli
from setuptools.config import read_configuration
from packaging.requirements import Requirement
from packaging.version import Version
from packaging.utils import canonicalize_name, canonicalize_version


parser = argparse.ArgumentParser(description="Check version strings for consistency")
parser.add_argument(
    "--tag_ref",
    type=str,
    default=None,
    help="The tag reference being deployed (if any)",
)


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


def find_build_macro_defs(setup_py):
    with open(setup_py, "r", encoding="utf8") as setup_file:
        content = setup_file.read()
    macros_re = re.compile(
        r"Extension\(.*?define_macros\s*=\s*(?P<defs>\[.+?\]).*?\)", re.DOTALL
    )
    if match := macros_re.search(content):
        return dict(ast.literal_eval(match.group("defs")))
    else:
        raise ValueError("Could not find build macro definitions")


def find_package_version(setup_cfg):
    cfg_file = read_configuration(setup_cfg)
    ver_str = str(cfg_file["metadata"]["version"])
    if ver_str != ver_str.strip():
        raise ValueError(f"Extra spaces in version string: '{ver_str}'")
    # Validate version format by constructing Version object
    return str(Version(ver_str))


def find_release_tag_version(tag_string):
    if tag_string is None:
        return None
    ver_re = re.compile(r"^refs/tags/v(?P<ver>.+)$")
    if match := ver_re.match(tag_string):
        return match.group("ver")
    else:
        raise ValueError(f"Invalid tag format {tag_string}")


def find_meta_min_python(setup_cfg):
    cfg_file = read_configuration(setup_cfg)
    ver_constraint = str(cfg_file["options"]["python_requires"])
    return find_min_version("python", ["python" + ver_constraint])


def find_macro_min_python(setup_py):
    macros = find_build_macro_defs(setup_py)
    min_python = macros["Py_LIMITED_API"]
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


def find_cibuildwheel_min_python(pyproject_toml):
    ver_re = re.compile(r"cp3(?P<minor>\d+)")
    with open(pyproject_toml, "rb") as pyproj_file:
        defs = tomli.load(pyproj_file)
    build_versions = defs["tool"]["cibuildwheel"]["build"]
    if not isinstance(build_versions, list):
        build_versions = build_versions.split()
    versions = []
    for ver in build_versions:
        if match := ver_re.match(ver):
            versions.append(f"python==3.{match.group('minor')}")
    return find_min_version("python", versions)


def find_package_min_numpy(setup_cfg):
    cfg_file = read_configuration(setup_cfg)
    ver_constraint = cfg_file["options"]["install_requires"]
    return find_min_version("numpy", filter(bool, map(str, ver_constraint)))


def find_setup_numpy_api(setup_py):
    macros = find_build_macro_defs(setup_py)
    min_numpy = macros["NPY_NO_DEPRECATED_API"]
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
    var_version = find_package_version("setup.cfg")
    tag_version = find_release_tag_version(args.tag_ref)
    print(f"Package variable version: {var_version}")
    if tag_version is not None:
        print(f"Release tag version: {tag_version}")
    # Check consistency
    if tag_version is not None and tag_version != var_version:
        print("Package version mismatch")
        failure = True
    print("")

    # Check Python version requirements
    meta_min_python = find_meta_min_python("setup.cfg")
    macro_limited_api = find_macro_min_python("setup.py")
    cibuildwheel_min_python = find_cibuildwheel_min_python("pyproject.toml")
    print(f"Metadata min Python: {meta_min_python}")
    print(f"Limited API macro: {macro_limited_api}")
    print(f"CIBW min Python: {cibuildwheel_min_python}")
    # Check consistency
    if not (meta_min_python == macro_limited_api == cibuildwheel_min_python):
        print("Python version mismatch")
        failure = True
    print("")

    # Check NumPy version requirements
    package_min_numpy = find_package_min_numpy("setup.cfg")
    macro_min_numpy = find_setup_numpy_api("setup.py")
    print(f"Package min NumPy: {package_min_numpy}")
    print(f"Macro min NumPy: {macro_min_numpy}")
    if package_min_numpy != macro_min_numpy:
        print("NumPy version mismatch")
        failure = True

    # Make sure versions match
    exit(1 if failure else 0)
