from setuptools import setup, find_packages, Extension
import numpy
import glob
import re


adrt_c_ext = Extension(
    "adrt._adrt_cdefs",
    sources=glob.glob("adrt/*.cpp"),
    depends=glob.glob("adrt/*.hpp"),
    language="c++",
    include_dirs=[numpy.get_include()],
    py_limited_api=True,
    define_macros=[
        ("Py_LIMITED_API", "0x03060000"),
        ("NPY_NO_DEPRECATED_API", "NPY_1_16_API_VERSION"),
    ],
)


def find_version(version_path):
    ver_re = re.compile(r"^__version__\s*=\s*['\"](?P<ver>.+?)['\"]")
    with open(version_path, mode="r", encoding="utf8") as version_file:
        for line in version_file:
            ver_match = ver_re.match(line)
            if ver_match:
                return ver_match.group("ver")
        raise ValueError("Could not find package version")


def load_readme(readme_path):
    with open(readme_path, mode="r", encoding="utf8") as readme_file:
        return readme_file.read()


setup(
    name="adrt",
    description="Fast approximate discrete Radon transform for NumPy arrays",
    version=find_version("adrt/__init__.py"),
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=["numpy>=1.16"],
    license="BSD",
    ext_modules=[adrt_c_ext],
    zip_safe=False,
    url="https://github.com/karlotness/adrt",
    long_description=load_readme("README.md"),
    long_description_content_type="text/markdown",
)
