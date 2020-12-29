# Project information
project = "adrt"
copyright = "2020, Karl Otness, Donsub Rim"
author = "Karl Otness, Donsub Rim"

# Other configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]

# Theme
html_theme = "sphinx_rtd_theme"

# Autodoc configuration
autodoc_mock_imports = []

# Napoleon configuration
napoleon_google_docstring = False

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Matplotlib plot directive configuration
plot_rcparams = {
    "figure.autolayout": True,
}

plot_pre_code = """
import numpy as np
from matplotlib import pyplot as plt
import adrt
"""
