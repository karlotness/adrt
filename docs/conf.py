import os
import sys

# Project information
project = "adrt"
copyright = "2020, Karl Otness, Donsub Rim"
author = "Karl Otness, Donsub Rim"

# Configure path to ADRT code
sys.path.insert(0, os.path.abspath(".."))

# Other configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]

# Theme
html_theme = "sphinx_rtd_theme"

# Autodoc configuration
autodoc_mock_imports = ["adrt._adrt_cdefs"]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
