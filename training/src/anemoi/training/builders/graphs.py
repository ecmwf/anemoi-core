# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from anemoi.models.utils.config import get_multiple_datasets_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.training.config_types import Settings


def _override_graph_dataset(graph_config: Mapping, dataset_path: str) -> None:
    if hasattr(graph_config.nodes, "data") and hasattr(graph_config.nodes.data.node_builder, "dataset"):
        graph_config.nodes.data.node_builder.dataset = dataset_path


def _create_graph_for_dataset(
    config: Settings,
    *,
    dataset_path: str,
    dataset_name: str,
) -> HeteroData:
    graph_filename = config.system.input.graph
    if graph_filename is not None:
        graph_filename = Path(graph_filename)
        if graph_filename.name.endswith(".pt"):
            graph_name = graph_filename.name.replace(".pt", f"_{dataset_name}.pt")
            graph_filename = graph_filename.parent / graph_name

        if graph_filename.exists() and not config.graph.overwrite:
            from anemoi.graphs.utils import get_distributed_device

            return torch.load(graph_filename, map_location=get_distributed_device(), weights_only=False)

    from anemoi.graphs.create import GraphCreator

    graph_config = config.graph
    _override_graph_dataset(graph_config, dataset_path)

    return GraphCreator(config=graph_config).create(
        save_path=graph_filename,
        overwrite=config.graph.overwrite,
    )


def build_graphs_from_config(
    config: Settings,
) -> dict[str, HeteroData]:
    """Build graph data for each dataset using the dataloader config."""
    graphs: dict[str, HeteroData] = {}
    dataset_configs = get_multiple_datasets_config(config.dataloader.training)
    for dataset_name, dataset_config in dataset_configs.items():
        graphs[dataset_name] = _create_graph_for_dataset(
            config,
            dataset_path=dataset_config["dataset"],
            dataset_name=dataset_name,
        )
    return graphs


__all__ = ["build_graphs_from_config"]
