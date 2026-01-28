# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING

from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.training.builders.components import build_component

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.training.config_types import Settings
    from anemoi.training.data.grid_indices import BaseGridIndices


def build_grid_indices_from_config(
    config: Settings,
    *,
    graph_data: Mapping[str, HeteroData],
) -> dict[str, BaseGridIndices]:
    """Build grid indices mapping from configuration."""
    grid_indices_config = get_multiple_datasets_config(config.dataloader.grid_indices)
    grid_indices: dict[str, BaseGridIndices] = {}

    for dataset_name, grid_config in grid_indices_config.items():
        grid_indices_instance = build_component(
            grid_config,
            reader_group_size=config.dataloader.read_group_size,
        )
        grid_indices_instance.setup(graph_data[dataset_name])
        grid_indices[dataset_name] = grid_indices_instance

    return grid_indices
