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
from typing import Any

from anemoi.training.builders.components import build_component

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.training.config_types import Settings


def build_output_masks_from_config(
    config: Settings,
    *,
    graph_data: Mapping[str, HeteroData],
) -> dict[str, Any]:
    """Build output masks for each dataset using the same config."""
    output_mask_config = config.model.output_mask
    if hasattr(output_mask_config, "model_dump"):
        output_mask_config = output_mask_config.model_dump(by_alias=True)

    return {
        dataset_name: build_component(output_mask_config, graph_data=dataset_graph)
        for dataset_name, dataset_graph in graph_data.items()
    }
