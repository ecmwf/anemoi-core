# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.graphs.create import GraphCreator
from anemoi.models.utils.config import get_multiple_datasets_config
from anemoi.models.utils.projection_helpers import uses_fused_dataset_graph
from anemoi.training.utils.graph_config import merge_projection_and_graph_config

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class TrainerGraphDataFactory:
    """Build the final trainer graph from config or an existing file.

    This factory keeps the trainer graph steps in one place:
    - detect load-only `graph=existing` mode
    - reuse the configured graph file when possible
    - expand projection configs before graph creation
    - choose between single-dataset and combined multi-dataset graph creation
    """

    def __init__(self, config: DictConfig) -> None:
        """Store the fully composed trainer config used to resolve graph inputs."""
        self.config = config

    @staticmethod
    def load_graph_from_file(graph_filename: Path) -> HeteroData:
        """Load a serialized graph on the currently active distributed device."""
        from anemoi.graphs.utils import get_distributed_device

        LOGGER.info("Loading graph data from %s", graph_filename)
        return torch.load(graph_filename, map_location=get_distributed_device(), weights_only=False)

    def is_existing_graph_mode(self) -> bool:
        """Return whether the trainer should load `system.input.graph` as-is.

        Existing-graph mode is selected when the user points at a graph file,
        disables overwrite, and the graph config itself does not define nodes or
        edges to build.
        """
        nodes = getattr(self.config.graph, "nodes", None)
        edges = getattr(self.config.graph, "edges", None)
        return (
            self.config.system.input.graph is not None and not self.config.graph.overwrite and not nodes and not edges
        )

    def existing_graph_path(self) -> Path:
        """Return the configured existing graph path, raising if it is missing."""
        graph_filename = Path(self.config.system.input.graph)
        if not graph_filename.exists():
            msg = f"Existing graph file not found: {graph_filename}"
            raise FileNotFoundError(msg)
        return graph_filename

    @staticmethod
    def validate_loaded_graph(graph_data: HeteroData, dataset_names: list[str]) -> None:
        """Ensure a loaded combined graph contains the dataset nodes required by training."""
        missing_nodes = [dataset_name for dataset_name in dataset_names if dataset_name not in graph_data.node_types]
        if missing_nodes:
            msg = (
                "Loaded graph is missing dataset node types required by the dataloader. "
                f"Missing {missing_nodes}; available nodes are {graph_data.node_types}."
            )
            raise ValueError(msg)

    def create_graph_for_dataset(self, dataset_path: str, dataset_name: str) -> HeteroData:
        """Create or load a single-dataset graph, including any derived projections."""
        graph_filename = None
        if self.config.system.input.graph is not None:
            graph_filename = Path(self.config.system.input.graph)
            if graph_filename.exists() and not self.config.graph.overwrite:
                return self.load_graph_from_file(graph_filename)

        graph_config = OmegaConf.create(OmegaConf.to_container(self.config.graph, resolve=False))
        merge_projection_and_graph_config(graph_config, dataset_names=[dataset_name])

        if hasattr(graph_config.nodes, "data") and hasattr(graph_config.nodes.data.node_builder, "dataset"):
            graph_config.nodes.data.node_builder.dataset = dataset_path

        return GraphCreator(config=graph_config).create(
            save_path=graph_filename,
            overwrite=self.config.graph.overwrite,
        )

    def create_fused_graph(self, dataset_names: list[str]) -> HeteroData:
        """Create or load one graph that already contains all dataset node groups."""
        graph_filename = None
        if self.config.system.input.graph is not None:
            graph_filename = Path(self.config.system.input.graph)
            if graph_filename.exists() and not self.config.graph.overwrite:
                return self.load_graph_from_file(graph_filename)

        graph_config = OmegaConf.create(OmegaConf.to_container(self.config.graph, resolve=False))
        merge_projection_and_graph_config(graph_config, dataset_names=dataset_names)

        return GraphCreator(config=graph_config).create(
            save_path=graph_filename,
            overwrite=self.config.graph.overwrite,
        )

    def build(self) -> HeteroData:
        """Return the graph object the trainer should hand to the model stack."""
        dataset_configs = get_multiple_datasets_config(self.config.dataloader.training)
        dataset_names = list(dataset_configs.keys())

        if self.is_existing_graph_mode():
            graph_data = self.load_graph_from_file(self.existing_graph_path())
            self.validate_loaded_graph(graph_data, dataset_names)
            return graph_data

        if uses_fused_dataset_graph(self.config.graph, dataset_names):
            LOGGER.info("Creating fused graph for datasets %s", dataset_names)
            return self.create_fused_graph(dataset_names)

        if len(dataset_names) != 1:
            msg = (
                "Multiple datasets require a fused graph config with one node group per dataset. "
                f"Received datasets {dataset_names} but graph nodes {list(self.config.graph.nodes.keys())}."
            )
            raise ValueError(msg)

        dataset_name = dataset_names[0]
        dataset_reader_config = dataset_configs[dataset_name].dataset_config
        if isinstance(dataset_reader_config, (DictConfig, dict)):
            if "dataset" not in dataset_reader_config:
                msg = f"Dataset '{dataset_name}' is missing 'dataset' key."
                raise ValueError(msg)
            dataset_source = dataset_reader_config["dataset"]
        else:
            dataset_source = dataset_reader_config

        if dataset_source is None:
            msg = f"Dataset source is None for dataset '{dataset_name}'. Check dataloader.dataset_config.dataset."
            raise ValueError(msg)

        LOGGER.info("Creating graph for dataset '%s'", dataset_name)
        return self.create_graph_for_dataset(dataset_source, dataset_name)
