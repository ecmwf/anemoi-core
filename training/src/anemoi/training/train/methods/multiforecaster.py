# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0

"""Compatibility task for multi-dataset training.

Some configs refer to `GraphMultiDatasetForecaster`. In this branch we can
reuse the standard `GraphForecaster` implementation because the base classes
already handle dicts of datasets.
"""

from __future__ import annotations

from anemoi.training.train.tasks.forecaster import GraphForecaster


class GraphMultiDatasetForecaster(GraphForecaster):
    """Alias of :class:`~anemoi.training.train.tasks.forecaster.GraphForecaster`."""
