"""Sphinx configuration."""
project = "Robotics Optimization Benchmarks"
author = "Charles Dawson"
copyright = "2023, Charles Dawson"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
