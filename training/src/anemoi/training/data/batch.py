# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Backward-compatible re-export of the :class:`Batch` envelope.

The canonical home of :class:`Batch` is :mod:`anemoi.models.data.batch`. It
lives there so the model layer can construct one without depending on
``anemoi-training``. This module re-exports the public symbols so existing
imports of the form ``from anemoi.training.data.batch import Batch`` keep
working.

Eventually the dataclass will be promoted to ``anemoi-utils``; this file
will then be turned into a deprecation shim.
"""

from __future__ import annotations

from anemoi.models.data.batch import BOUNDARIES_META_KEY
from anemoi.models.data.batch import STATIC_COORDS_META_KEY
from anemoi.models.data.batch import Batch
from anemoi.models.data.batch import SourceView
from anemoi.models.data.batch import TensorLayout

__all__ = ["BOUNDARIES_META_KEY", "STATIC_COORDS_META_KEY", "Batch", "SourceView", "TensorLayout"]
