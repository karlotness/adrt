import argparse
import re
import ast
import configparser
import toml

parser = argparse.ArgumentParser(description="Check version strings for consistency")
parser.add_argument(
    "--tag_ref",
    type=str,
    default=None,
    help="The tag reference being deployed (if any)",
)


def find_build_macro_defs(setup_py):
    with open(setup_py, "r", encoding="utf8") as setup_file:
        content = setup_file.read()
    macros_re = re.compile(
        r"Extension\(.*?define_macros\s*=\s*(?P<defs>\[.+?\]).*?\)", re.DOTALL
    )
    match = macros_re.search(content)
    if not match:
        raise ValueError("Could not find macro definitions")
    return dict(ast.literal_eval(match.group("defs")))


def find_package_meta_version(setup_cfg):
    cfg_file = configparser.ConfigParser()
    cfg_file.read(setup_cfg)
    return cfg_file["metadata"]["version"]


def find_package_var_version(version_path):
    ver_re = re.compile(r"^__version__\s*=\s*['\"](?P<ver>.+?)['\"]")
    with open(version_path, mode="r", encoding="utf8") as version_file:
        for line in version_file:
            ver_match = ver_re.match(line)
            if ver_match:
                return ver_match.group("ver")
        raise ValueError("Could not find package version")


def find_release_tag_version(tag_string):
    if tag_string is None:
        return None
    ver_re = re.compile(r"^refs/tags/v(?P<ver>.+)$")
    match = ver_re.match(tag_string)
    if not match:
        raise ValueError(f"Invalid tag format {tag_string}")
    return match.group("ver")


def find_meta_min_python(setup_cfg):
    cfg_file = configparser.ConfigParser()
    cfg_file.read(setup_cfg)
    ver_constraint = cfg_file["options"]["python_requires"]
    ver_re = re.compile(r">=\s*(?P<ver>.+?)\s*,\s*")
    match = ver_re.match(ver_constraint)
    if not match:
        raise ValueError("Could not parse constraint: {ver_constraint}")
    return match.group("ver")


def find_wheel_limited_api(setup_cfg):
    cfg_file = configparser.ConfigParser()
    cfg_file.read(setup_cfg)
    limited_ver = cfg_file["bdist_wheel"]["py-limited-api"].strip()
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
    ver = f"{major}.{minor}"
    if micro != 0 or release != 0:
        ver += f".{micro}"
    if release != 0:
        ver += f"{hex(release)[2:]}"
    return ver


def find_package_min_numpy(setup_cfg):
    cfg_file = configparser.ConfigParser()
    cfg_file.read(setup_cfg)
    ver_constraint = cfg_file["options"]["install_requires"].split()
    for constaint in ver_constraint:
        constaint = constaint.strip()
        if constaint.startswith("numpy") and constaint[5:7] == ">=":
            return constaint[7:]
    raise ValueError("Could not find minimum NumPy version")


def find_pyproject_min_numpy(pyproject_toml):
    with open(pyproject_toml, "r", encoding="utf8") as pyproj_file:
        defs = toml.load(pyproj_file)
    for constaint in defs["build-system"]["requires"]:
        constaint = constaint.strip()
        if constaint.startswith("numpy") and constaint[5:7] == ">=":
            return constaint[7:]
    raise ValueError("Could not find minimum NumPy version")


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
    meta_version = find_package_meta_version("setup.cfg")
    var_version = find_package_var_version("src/adrt/__init__.py")
    tag_version = find_release_tag_version(args.tag_ref)
    print(f"Metadata version: {meta_version}")
    print(f"Package variable version: {var_version}")
    if tag_version is not None:
        print(f"Release tag version: {tag_version}")
    # Check consistency
    if (meta_version != var_version) or (
        tag_version is not None and tag_version != var_version
    ):
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
    print(f"Package min NumPy: {package_min_numpy}")
    print(f"PyProject min NumPy: {pyproj_min_numpy}")
    print(f"Macro min NumPy: {macro_min_numpy}")
    if not (package_min_numpy == pyproj_min_numpy == macro_min_numpy):
        print("NumPy version mismatch")
        failure = True

    # Make sure versions match
    if failure:
        exit(1)
    else:
        exit(0)
