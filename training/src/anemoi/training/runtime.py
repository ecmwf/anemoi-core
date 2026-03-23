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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class TaskRuntimeArtifacts:
    """Data prepared by the trainer and passed to the Lightning task."""

    graph_data: HeteroData
    statistics: dict
    statistics_tendencies: dict | None
    data_indices: dict
    metadata: dict
    supporting_arrays: dict


@dataclass(frozen=True)
class ModelRuntimeArtifacts:
    """Data prepared by the trainer and passed when creating the model."""

    graph_data: HeteroData
    statistics: dict
    statistics_tendencies: dict | None
    data_indices: dict
    metadata: dict
    supporting_arrays: dict

    def to_task_runtime_artifacts(self) -> TaskRuntimeArtifacts:
        """Return the same data in the form used by tasks."""
        return TaskRuntimeArtifacts(
            graph_data=self.graph_data,
            statistics=self.statistics,
            statistics_tendencies=self.statistics_tendencies,
            data_indices=self.data_indices,
            metadata=self.metadata,
            supporting_arrays=self.supporting_arrays,
        )
