# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Helpers for detecting fused multi-dataset graphs."""

from __future__ import annotations

DEFAULT_DATASET_NAME = "data"
DEFAULT_EDGE_RELATION_NAME = "to"
DEFAULT_EDGE_WEIGHT_ATTRIBUTE = "gauss_weight"
DEFAULT_GAUSSIAN_NORM = "l1"
