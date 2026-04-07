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

from omegaconf import DictConfig

from anemoi.graphs.factory import GraphDataFactory
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.graphs.projections import ProjectionCreator
from anemoi.graphs.projections import ProjectionData
from anemoi.models.utils.config import get_multiple_datasets_config

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class TrainerGraphDataFactory:
    """Build the final trainer graph from config or an existing file.

    Wraps ``GraphDataFactory`` from ``anemoi.graphs`` and additionally handles
    training-specific concerns: detecting load-only mode, resolving dataset
    source paths from the dataloader config, and producing per-dataset
    ``ProjectionData`` for loss and model instantiation.
    """

    def __init__(self, config: DictConfig) -> None:
        """Store the fully composed trainer config used to resolve graph inputs."""
        self.config = config
        self._graph_factory = GraphDataFactory(config.graph)

    @staticmethod
    def load_graph_from_file(graph_filename: Path) -> HeteroData:
        """Load a serialized graph on the currently active distributed device."""
        return GraphDataFactory.load_graph_from_file(graph_filename)

    def is_existing_graph_mode(self) -> bool:
        """Return whether the trainer should load ``system.input.graph`` as-is."""
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
        GraphDataFactory.validate_loaded_graph(graph_data, dataset_names)

    def _required_existing_graph_dataset_names(self, dataset_names: list[str]) -> list[str]:
        """Return dataset node names required when loading a graph as-is."""
        if len(dataset_names) > 1:
            return dataset_names
        return [DEFAULT_DATASET_NAME]

    def _graph_output_path(self) -> Path | None:
        graph_path = self.config.system.input.graph
        return Path(graph_path) if graph_path is not None else None

    def _resolve_dataset_path(self, dataset_names: list[str], dataset_configs: dict) -> str | None:
        """Return the dataset source path for single-dataset non-fused graphs."""
        if uses_fused_dataset_graph(self.config.graph, dataset_names):
            return None
        if len(dataset_names) != 1:
            msg = (
                "Multiple datasets require a fused graph config with one node group per dataset. "
                f"Received datasets {dataset_names} but graph nodes "
                f"{list(self.config.graph.nodes.keys())}."
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
            msg = f"Dataset source is None for dataset '{dataset_name}'."
            raise ValueError(msg)
        return dataset_source

    def _resolve_projection_data_per_dataset(
        self,
        graph_data: HeteroData,
        dataset_names: list[str],
    ) -> dict[str, ProjectionData]:
        """Build a ``ProjectionData`` for each dataset."""
        projections_cfg = self.config.graph.get("projections") or {}
        creator = ProjectionCreator(config=projections_cfg)
        return {
            dataset_name: creator.create(
                graph_data,
                [dataset_name] if not uses_fused_dataset_graph(self.config.graph, dataset_names) else dataset_names,
            )
            for dataset_name in dataset_names
        }

    def build(self) -> tuple[HeteroData, dict[str, ProjectionData]]:
        """Return the graph and per-dataset projection metadata.

        Returns
        -------
        graph_data:
            Built or loaded ``HeteroData`` graph.
        projection_data:
            Dict mapping each dataset name to its resolved ``ProjectionData``.
        """
        dataset_configs = get_multiple_datasets_config(self.config.dataloader.training)
        dataset_names = list(dataset_configs.keys())

        if self.is_existing_graph_mode():
            graph_data = self.load_graph_from_file(self.existing_graph_path())
            self.validate_loaded_graph(graph_data, self._required_existing_graph_dataset_names(dataset_names))
            projection_data = self._resolve_projection_data_per_dataset(graph_data, dataset_names)
            return graph_data, projection_data

        LOGGER.info(
            "Creating %s graph for datasets %s",
            "fused" if uses_fused_dataset_graph(self.config.graph, dataset_names) else "single-dataset",
            dataset_names,
        )

        dataset_path = self._resolve_dataset_path(dataset_names, dataset_configs)
        graph_data, _ = self._graph_factory.build(
            dataset_names=dataset_names,
            dataset_path=dataset_path,
            save_path=self._graph_output_path(),
        )
        projection_data = self._resolve_projection_data_per_dataset(graph_data, dataset_names)
        return graph_data, projection_data
