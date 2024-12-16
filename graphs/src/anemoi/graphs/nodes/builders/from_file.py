# (C) Copyright 2024 Anemoi contributors.
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

import numpy as np
import torch
from anemoi.datasets import open_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class ZarrDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset.

    Attributes
    ----------
    dataset : str | DictConfig
        The dataset.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attrs_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, dataset: DictConfig, name: str) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.dataset = dataset if isinstance(dataset, str) else OmegaConf.to_container(dataset)
        super().__init__(name)
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"dataset"}

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        dataset = open_dataset(self.dataset)
        return self.reshape_coords(dataset.latitudes, dataset.longitudes)


class TextNodes(BaseNodeBuilder):
    """Nodes from text file.

    Attributes
    ----------
    dataset : str | DictConfig
        The path to txt file containing the coordinates of the nodes.
    idx_lon : int
        The index of the longitude in the dataset.
    idx_lat : int
        The index of the latitude in the dataset.
    """

    def __init__(self, dataset, name: str, idx_lon: int = 0, idx_lat: int = 1) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.dataset = np.loadtxt(dataset)
        self.idx_lon = idx_lon
        self.idx_lat = idx_lat
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        return self.reshape_coords(self.dataset[self.idx_lat, :], self.dataset[self.idx_lon, :])


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
    update_graph(graph, name, attrs_config)
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

        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(resolution, grid_definition_path, name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)

    def get_coordinates(self) -> np.ndarray:
        coords = super().get_coordinates()

        LOGGER.info(
            "Limiting the processor mesh to a radius of %.2f km from the output mesh.",
            self.area_mask_builder.margin_radius_km,
        )
        area_mask = self.area_mask_builder.get_mask(coords)

        LOGGER.info(
            "Dropping %d nodes from the processor mesh.",
            len(area_mask) - area_mask.sum(),
        )
        coords = coords[area_mask]

        return coords
