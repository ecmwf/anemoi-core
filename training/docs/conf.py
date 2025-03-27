# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys
from pathlib import Path

read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

sys.path.insert(0, Path("..").absolute() / "src")


source_suffix = ".rst"
master_doc = "index"
pygments_style = "sphinx"
html_theme_options = {"logo_only": True}
html_logo = "_static/logo.png"


# -- Project information -----------------------------------------------------

project = "Anemoi Training"

author = "Anemoi contributors"

year = datetime.datetime.now(tz=datetime.timezone.utc).year
years = "2024" if year == 2024 else f"2024-{year}"

copyright = f"{years}, Anemoi contributors"  # noqa: A001

try:
    from anemoi.training._version import __version__

    release = __version__
except ImportError:
    release = "0.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "sphinx.ext.autosectionlabel",
    # "sphinxcontrib.autodoc_pydantic",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"] # noqa: ERA001

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "'**.ipynb_checkpoints'"]

intersphinx_mapping = {
    "python": ("https://python.readthedocs.io/en/latest", None),
    "anemoi-docs": (
        "https://anemoi.readthedocs.io/en/latest/",
        ("../../../anemoi-docs/docs/_build/html/objects.inv", None),
    ),
    "anemoi-utils": (
        "https://anemoi-utils.readthedocs.io/en/latest/",
        ("../../../anemoi-utils/docs/_build/html/objects.inv", None),
    ),
    "anemoi-datasets": (
        "https://anemoi-datasets.readthedocs.io/en/latest/",
        ("../../../anemoi-datasets/docs/_build/html/objects.inv", None),
    ),
    "anemoi-models": (
        "https://anemoi-models.readthedocs.io/en/latest/",
        ("../../anemoi-models/docs/_build/html/objects.inv", None),
    ),
    "anemoi-training": (
        "https://anemoi-training.readthedocs.io/en/latest/",
        ("../../anemoi-training/docs/_build/html/objects.inv", None),
    ),
    "anemoi-inference": (
        "https://anemoi-inference.readthedocs.io/en/latest/",
        ("../../../anemoi-inference/docs/_build/html/objects.inv", None),
    ),
    "anemoi-graphs": (
        "https://anemoi-graphs.readthedocs.io/en/latest/",
        ("../../anemoi-graphs/docs/_build/html/objects.inv", None),
    ),
    "anemoi-registry": (
        "https://anemoi-registry.readthedocs.io/en/latest/",
        ("../../../anemoi-registry/docs/_build/html/objects.inv", None),
    ),
    "anemoi-transform": (
        "https://anemoi-transform.readthedocs.io/en/latest/",
        ("../../../anemoi-transform/docs/_build/html/objects.inv", None),
    ),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["style.css"]


todo_include_todos = not read_the_docs_build

autodoc_member_order = "bysource"  # Keep file order


# https://autodoc-pydantic.readthedocs.io/en/stable/users/configuration.html

autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"
