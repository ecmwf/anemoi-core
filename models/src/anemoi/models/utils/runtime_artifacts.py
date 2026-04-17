# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class RuntimeArtifacts:
    """Pure runtime data passed from AnemoiTrainer to the model builder.

    All fields are plain dicts — no config objects.
    """

    statistics: dict  # keyed by dataset name
    data_indices: dict  # keyed by dataset name
    supporting_arrays: dict  # raw from datamodule; combined with graph masks inside builder
    graph_data: dict
    metadata: dict
    statistics_tendencies: dict | None = field(default=None)  # keyed by dataset name
