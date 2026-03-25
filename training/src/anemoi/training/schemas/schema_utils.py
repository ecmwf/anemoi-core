# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Re-export facade — import from schema_defaults or interpolation_anchors directly."""

from .interpolation_anchors import prune_undeclared_interpolation_anchors
from .interpolation_anchors import resolve_and_prune_undeclared_interpolation_anchors
from .interpolation_anchors import undeclared_interpolation_anchor_paths
from .schema_defaults import DatasetDict
from .schema_defaults import apply_schema_defaults
from .schema_defaults import schema_defaults

__all__ = [
    "DatasetDict",
    "apply_schema_defaults",
    "prune_undeclared_interpolation_anchors",
    "resolve_and_prune_undeclared_interpolation_anchors",
    "schema_defaults",
    "undeclared_interpolation_anchor_paths",
]
