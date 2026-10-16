# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Hyperparameters stored in the training checkpoint, and which operations need them.

The hyperparameters are the ``__init__`` arguments of ``BaseTrainingModule``. They can be left out of the
checkpoint with ``diagnostics.checkpoint.exclude_hyperparameters``.
"""

# All ``__init__`` arguments of ``BaseTrainingModule``
INIT_HYPERPARAMETERS = frozenset(
    {
        "config",
        "task",
        "graph_data",
        "statistics",
        "statistics_tendencies",
        "data_indices",
        "metadata",
        "supporting_arrays",
    },
)

# Hyperparameters required for restoring a training checkpoint (resume, load weights only, transfer learning):
HYPERPARAMETERS_FOR_RESUME = frozenset({"config", "data_indices", "metadata"})

# Hyperparameters required for ``anemoi-training checkpoint inference``
HYPERPARAMETERS_FOR_INFERENCE_CONVERSION = INIT_HYPERPARAMETERS
