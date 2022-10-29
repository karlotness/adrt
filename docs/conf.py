import sys
import shutil
import pathlib
import inspect
import importlib
import functools
import itertools
import re
import packaging.version
import adrt

# Project information
project = "adrt"
copyright = "2022 Karl Otness, Donsub Rim"
author = "Karl Otness, Donsub Rim"
version = adrt.__version__
release = version

# Other configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx.ext.linkcode",
    "sphinx_rtd_theme",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
suppress_warnings = ["epub.unknown_project_files"]

# Insert code into each rst file
rst_prolog = r"""

.. role:: pycode(code)
   :language: python

.. role:: cppcode(code)
   :language: cpp

"""

# Theme
html_theme = "sphinx_rtd_theme"

# Autodoc configuration
autodoc_mock_imports = []
autodoc_typehints = "none"

# Napoleon configuration
napoleon_google_docstring = False

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "pypug": ("https://packaging.python.org/en/latest/", None),
}

# Matplotlib plot directive configuration
plot_formats = [("png", 100), ("hires.png", 200)]
plot_rcparams = {
    "savefig.bbox": "tight",
}
plot_apply_rcparams = True
plot_include_source = True
plot_html_show_source_link = False
plot_pre_code = """
import numpy as np
from matplotlib import pyplot as plt
import adrt
"""


# Image converter settings
def adrt_magick_available():
    if sys.platform == "win32":
        return shutil.which("magick") is not None
    return shutil.which("convert") is not None


if not adrt_magick_available():
    extensions.remove("sphinx.ext.imgconverter")


# Linkcode configuration
@functools.lru_cache(maxsize=1)
def adrt_find_anchors():
    repo_root = pathlib.Path(__file__).parent.resolve().parent
    source_root = repo_root / "src" / "adrt"
    if not (source_root.exists() and source_root.is_dir()):
        return {}
    anchor_re = re.compile(
        r"^\s*//\s*DOC ANCHOR:\s+(?P<name>\S+)(?:\s+\+\s*(?P<offset>\d+))?\s*$"
    )
    anchor_map = {}
    for source_path in itertools.chain(
        source_root.glob("*.cpp"), source_root.glob("*.hpp")
    ):
        if not source_path.exists():
            continue
        with open(source_path, "r", encoding="utf8") as source_file:
            for lnum, line in enumerate(source_file, start=1):
                if re_match := anchor_re.match(line):
                    anchor_name = re_match.group("name").strip()
                    if re_match.group("offset") is not None:
                        anchor_offset = int(re_match.group("offset"))
                    else:
                        anchor_offset = 1
                    anchor_map[anchor_name] = (source_path.name, lnum + anchor_offset)
    return anchor_map


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    mod_name = info["module"]
    if mod_name != "adrt" and not mod_name.startswith("adrt."):
        return None
    fullname = info["fullname"]
    qualname = f"{mod_name}.{fullname}"
    link_name_map = adrt_find_anchors()
    if qualname in link_name_map:
        # Use the provided anchor
        source_file, line_start = link_name_map[qualname]
        line_end = None
    else:
        pkg_root = pathlib.Path(adrt.__file__).parent
        module = importlib.import_module(mod_name)
        obj = getattr(module, fullname)
        try:
            source_file = str(
                pathlib.Path(inspect.getsourcefile(obj)).relative_to(pkg_root)
            )
        except ValueError:
            return None
        lines, line_start = inspect.getsourcelines(obj)
        line_end = line_start + len(lines) - 1
    # Form the URL from the pieces
    repo_url = "https://github.com/karlotness/adrt"
    if packaging.version.Version(version).is_devrelease:
        ref = "master"
    else:
        ref = f"v{version}"
    if line_start and line_end:
        line_suffix = f"#L{line_start}-L{line_end}"
    elif line_start:
        line_suffix = f"#L{line_start}"
    else:
        line_suffix = ""
    return f"{repo_url}/blob/{ref}/src/adrt/{source_file}{line_suffix}"
