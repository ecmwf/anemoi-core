from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class ZarrDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset.

    Attributes
    ----------
    dataset : zarr.core.Array
        The dataset.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, dataset: DotDict, name: str) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.dataset = open_dataset(dataset)
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        return self.reshape_coords(self.dataset.latitudes, self.dataset.longitudes)


class CutOutZarrDatasetNodes(ZarrDatasetNodes):
    """Nodes from Zarr dataset."""

    def __init__(
        self, name: str, lam_dataset: str, forcing_dataset: str, thinning: int = 1, adjust: str = "all"
    ) -> None:
        dataset_config = {
            "cutout": [{"dataset": lam_dataset, "thinning": thinning}, {"dataset": forcing_dataset}],
            "adjust": adjust,
        }
        super().__init__(dataset_config, name)
        self.n_cutout, self.n_other = self.dataset.grids

    def register_attributes(self, graph: HeteroData, config: DotDict) -> None:
        # this is a mask to cutout the LAM area
        graph[self.name]["cutout"] = torch.tensor([True] * self.n_cutout + [False] * self.n_other, dtype=bool).reshape(
            (-1, 1)
        )
        return super().register_attributes(graph, config)


class NPZFileNodes(BaseNodeBuilder):
    """Nodes from NPZ defined grids.

    Attributes
    ----------
    resolution : str
        The resolution of the grid.
    grid_definition_path : str
        Path to the folder containing the grid definition files.
    grid_definition : dict[str, np.ndarray]
        The grid definition.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, resolution: str, grid_definition_path: str, name: str) -> None:
        """Initialize the NPZFileNodes builder.

        The builder suppose the grids are stored in files with the name `grid-{resolution}.npz`.

        Parameters
        ----------
        resolution : str
            The resolution of the grid.
        grid_definition_path : str
            Path to the folder containing the grid definition files.
        """
        self.resolution = resolution
        self.grid_definition_path = grid_definition_path
        self.grid_definition = np.load(Path(self.grid_definition_path) / f"grid-{self.resolution}.npz")
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords


class LimitedAreaNPZFileNodes(NPZFileNodes):
    """Nodes from NPZ defined grids using an area of interest."""

    def __init__(
        self,
        resolution: str,
        grid_definition_path: str,
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:

        self.aoi_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(resolution, grid_definition_path, name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.aoi_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> np.ndarray:
        coords = super().get_coordinates()

        LOGGER.info(
            "Limiting the processor mesh to a radius of %.2f km from the output mesh.",
            self.aoi_mask_builder.margin_radius_km,
        )
        aoi_mask = self.aoi_mask_builder.get_mask(coords)

        LOGGER.info("Dropping %d nodes from the processor mesh.", len(aoi_mask) - aoi_mask.sum())
        coords = coords[aoi_mask]

        return coords
