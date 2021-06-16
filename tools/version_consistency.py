# Copyright (c) 2020, 2021 Karl Otness, Donsub Rim
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
import configparser
import toml
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
    match = macros_re.search(content)
    if not match:
        raise ValueError("Could not find build macro definitions")
    return dict(ast.literal_eval(match.group("defs")))


def find_package_version(setup_cfg):
    cfg_file = read_configuration(setup_cfg)
    return str(cfg_file["metadata"]["version"])


def find_release_tag_version(tag_string):
    if tag_string is None:
        return None
    ver_re = re.compile(r"^refs/tags/v(?P<ver>.+)$")
    match = ver_re.match(tag_string)
    if not match:
        raise ValueError(f"Invalid tag format {tag_string}")
    return match.group("ver")


def find_meta_min_python(setup_cfg):
    cfg_file = read_configuration(setup_cfg)
    ver_constraint = str(cfg_file["options"]["python_requires"])
    return find_min_version("python", ["python" + ver_constraint])


def find_wheel_limited_api(setup_cfg):
    cfg_file = configparser.ConfigParser()
    cfg_file.read(setup_cfg)
    limited_ver = cfg_file["bdist_wheel"]["py_limited_api"].strip()
    if not limited_ver.startswith("cp"):
        raise ValueError(f"Could not parse limited API version: {limited_ver}")
    major = limited_ver[2]
    minor = limited_ver[3:]
    return f"{major}.{minor}"


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


def find_package_min_numpy(setup_cfg):
    cfg_file = read_configuration(setup_cfg)
    ver_constraint = cfg_file["options"]["install_requires"]
    return find_min_version("numpy", filter(bool, map(str, ver_constraint)))


def find_pyproject_min_numpy(pyproject_toml):
    with open(pyproject_toml, "r", encoding="utf8") as pyproj_file:
        defs = toml.load(pyproj_file)
    return find_min_version("numpy", defs["build-system"]["requires"])


def find_constraint_min_numpy(constraint_txt):
    comment_re = re.compile(r"(?:^|\s)#.+$")
    constraints = []
    with open(constraint_txt, "r", encoding="utf8") as constraint_file:
        for line in constraint_file:
            constraints.append(comment_re.sub("", line.strip()))
    return find_min_version("numpy", filter(bool, constraints))


def find_setup_numpy_api(setup_py):
    macros = find_build_macro_defs(setup_py)
    min_numpy = macros["NPY_NO_DEPRECATED_API"]
    rgx = re.compile(r"^NPY_(?P<major>\d+)_(?P<minor>\d+)_API_VERSION$")
    match = rgx.match(min_numpy)
    if not match:
        raise ValueError(f"Invalid NumPy API macro: {min_numpy}")
    major = match.group("major")
    minor = match.group("minor")
    return f"{major}.{minor}"


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
    wheel_limited_api = find_wheel_limited_api("setup.cfg")
    macro_limited_api = find_macro_min_python("setup.py")
    print(f"Metadata min Python: {meta_min_python}")
    print(f"Wheel limited API version: {wheel_limited_api}")
    print(f"Limited API macro: {macro_limited_api}")
    # Check consistency
    if not (meta_min_python == wheel_limited_api == macro_limited_api):
        print("Python version mismatch")
        failure = True
    print("")

    # Check NumPy version requirements
    package_min_numpy = find_package_min_numpy("setup.cfg")
    pyproj_min_numpy = find_pyproject_min_numpy("pyproject.toml")
    macro_min_numpy = find_setup_numpy_api("setup.py")
    constraint_min_numpy = find_constraint_min_numpy("tools/constraints.txt")
    print(f"Package min NumPy: {package_min_numpy}")
    print(f"PyProject min NumPy: {pyproj_min_numpy}")
    print(f"Macro min NumPy: {macro_min_numpy}")
    print(f"Constraints min NumPy: {constraint_min_numpy}")
    if not (
        package_min_numpy == pyproj_min_numpy == macro_min_numpy == constraint_min_numpy
    ):
        print("NumPy version mismatch")
        failure = True

    # Make sure versions match
    if failure:
        exit(1)
    else:
        exit(0)
