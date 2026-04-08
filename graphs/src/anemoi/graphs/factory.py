# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Factory for building graphs and resolving projection metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.graphs.projections import ProjectionCreator
from anemoi.graphs.projections import ProjectionData  # re-exported for callers

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class GraphDataFactory:
    """Build a graph and resolve projection metadata in one step.

    Mirrors ``GraphCreator`` but additionally runs ``ProjectionCreator`` after
    graph creation so callers receive both the graph and the resolved projection
    names they need to instantiate model layers and loss wrappers.

    Parameters
    ----------
    config:
        The ``graph`` section of the trainer config (``DictConfig`` or plain
        dict).  Must contain ``nodes``, ``edges``, and optionally
        ``projections`` and ``overwrite``.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        dataset_names: list[str],
        dataset_path: str | None = None,
        save_path: Path | None = None,
    ) -> tuple[HeteroData, dict[str, ProjectionData]]:
        """Load an existing graph or build a new one, then resolve projections.

        Parameters
        ----------
        dataset_names:
            Names of the dataset node groups expected in the graph.
        dataset_path:
            Path to the dataset source used to override the data node builder
            for single-dataset graphs.
        save_path:
            Where to save the graph.  When ``None`` the graph is not saved.

        Returns
        -------
        graph_data:
            The built (or loaded) ``HeteroData`` graph.
        projection_data:
            Per-dataset resolved projection metadata derived from the graph.
            For fused graphs every dataset resolves against the full
            ``dataset_names`` list; for non-fused graphs each dataset is
            resolved independently.
        """
        overwrite = self.config.get("overwrite", False)

        graph_data = self._load_existing_if_available(save_path, dataset_names)
        if graph_data is None:
            graph_config = self._build_graph_config(dataset_names, dataset_path=dataset_path)
            graph_data = GraphCreator(config=graph_config).create(
                save_path=save_path,
                overwrite=overwrite,
            )

        creator = ProjectionCreator(config=self.config.get("projections") or {})
        fused = uses_fused_dataset_graph(self.config, dataset_names)
        projection_data = {
            name: creator.create(graph_data, dataset_names if fused else [name]) for name in dataset_names
        }

        return graph_data, projection_data

    # ------------------------------------------------------------------
    # Graph loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_graph_from_file(graph_filename: Path) -> HeteroData:
        """Load a serialized graph on the currently active distributed device."""
        try:
            from anemoi.graphs.utils import get_distributed_device

            map_location = get_distributed_device()
        except Exception:
            map_location = "cpu"

        LOGGER.info("Loading graph data from %s", graph_filename)
        return torch.load(graph_filename, map_location=map_location, weights_only=False)

    @staticmethod
    def validate_loaded_graph(graph_data: HeteroData, required_dataset_names: list[str]) -> None:
        """Ensure the loaded graph contains the required dataset node types."""
        missing = [n for n in required_dataset_names if n not in graph_data.node_types]
        if missing:
            msg = (
                "Loaded graph is missing dataset node types required by the dataloader. "
                f"Missing {missing}; available nodes are {graph_data.node_types}."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing_if_available(
        self,
        save_path: Path | None,
        dataset_names: list[str],
    ) -> HeteroData | None:
        """Return the saved graph if overwrite is disabled and the file exists."""
        if save_path is None or self.config.get("overwrite", False) or not Path(save_path).exists():
            return None

        graph_data = self.load_graph_from_file(Path(save_path))
        required = dataset_names if uses_fused_dataset_graph(self.config, dataset_names) else [DEFAULT_DATASET_NAME]
        self.validate_loaded_graph(graph_data, required)
        return graph_data

    def _build_graph_config(
        self,
        dataset_names: list[str],
        dataset_path: str | None = None,
    ) -> DictConfig:
        """Clone the graph config, expand projections, inject dataset path."""
        from anemoi.graphs.graph_config import merge_projection_and_graph_config

        graph_config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=False))
        merge_projection_and_graph_config(graph_config, dataset_names=dataset_names)

        if dataset_path is not None and graph_config.get("nodes") is not None:
            data_node_cfg = graph_config.nodes.get(DEFAULT_DATASET_NAME)
            if (
                data_node_cfg is not None
                and hasattr(data_node_cfg, "node_builder")
                and hasattr(data_node_cfg.node_builder, "dataset")
            ):
                data_node_cfg.node_builder.dataset = dataset_path

        return graph_config
