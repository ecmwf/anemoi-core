# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Sphinx configuration for anemoi-metadata documentation."""

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

# Add source directory to path for autodoc
sys.path.insert(0, os.path.join(os.path.abspath(".."), "src"))

source_suffix = ".rst"
master_doc = "index"
pygments_style = "sphinx"

# -- Project information -----------------------------------------------------

project = "Anemoi Metadata"
author = "Anemoi contributors"

year = datetime.datetime.now().year
if year == 2024:
    years = "2024"
else:
    years = f"2024-{year}"

copyright = f"{years}, Anemoi contributors"

try:
    from anemoi.metadata._version import __version__

    release = __version__
except ImportError:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinxarg.ext",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"

# Autosummary settings
autosummary_generate = True

# Templates path
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "'**.ipynb_checkpoints'"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://python.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "anemoi-docs": (
        "https://anemoi.readthedocs.io/en/latest/",
        ("../../../anemoi-docs/docs/_build/html/objects.inv", None),
    ),
    "anemoi-utils": (
        "https://anemoi-utils.readthedocs.io/en/latest/",
        ("../../anemoi-utils/docs/_build/html/objects.inv", None),
    ),
    "anemoi-datasets": (
        "https://anemoi-datasets.readthedocs.io/en/latest/",
        ("../../anemoi-datasets/docs/_build/html/objects.inv", None),
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
        ("../../anemoi-inference/docs/_build/html/objects.inv", None),
    ),
    "anemoi-graphs": (
        "https://anemoi-graphs.readthedocs.io/en/latest/",
        ("../../anemoi-graphs/docs/_build/html/objects.inv", None),
    ),
    "anemoi-registry": (
        "https://anemoi-registry.readthedocs.io/en/latest/",
        ("../../anemoi-registry/docs/_build/html/objects.inv", None),
    ),
    "anemoi-transform": (
        "https://anemoi-transform.readthedocs.io/en/latest/",
        ("../../anemoi-transform/docs/_build/html/objects.inv", None),
    ),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {"logo_only": True}
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Todo extension
todo_include_todos = not read_the_docs_build

# -- Sphinx-pydantic settings ------------------------------------------------

# Enable automatic field documentation
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_summary_list_order = "bysource"
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_field_show_constraints = True
autodoc_pydantic_field_show_alias = True
autodoc_pydantic_field_show_default = True
