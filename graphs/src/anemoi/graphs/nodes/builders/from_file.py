# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.nodes.builders.base import BaseNodeBuilder

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetNodes(BaseNodeBuilder):
    """Nodes from an anemoi dataset.

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
        from anemoi.datasets import open_dataset

        dataset = open_dataset(self.dataset)
        return self.reshape_coords(dataset.latitudes, dataset.longitudes)


class TextNodes(BaseNodeBuilder):
    """Nodes from text file.

    Attributes
    ----------
    dataset : str | Path
        The path including filename to txt file containing the coordinates of the nodes.
    idx_lon : int
        The index of the longitude in the dataset.
    idx_lat : int
        The index of the latitude in the dataset.
    """

    def __init__(self, dataset: str | Path, name: str, idx_lon: int = 0, idx_lat: int = 1) -> None:
        LOGGER.info("Reading the dataset from %s.", dataset)
        self.dataset = dataset
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
        dataset = np.loadtxt(self.dataset)
        return self.reshape_coords(dataset[self.idx_lat, :], dataset[self.idx_lon, :])


class NPZFileNodes(BaseNodeBuilder):
    """Nodes from NPZ defined grids.

    Attributes
    ----------
    npz_file : str
        Path to the file.
    lat_key : str
        Name of the key of the latitude arrays.
    lon_key : str
        Name of the key of the latitude arrays.

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

    def __init__(
        self,
        npz_file: str,
        name: str,
        lat_key: str = "latitudes",
        lon_key: str = "longitudes",
    ) -> None:
        """Initialize the NPZFileNodes builder.

        The builder suppose the grids are stored in files with the name `grid-{resolution}.npz`.

        Parameters
        ----------
        npz_file : str
            The path to the file.
        name : str
            Name of the nodes to be added.
        lat_key : str, optional
            Name of the key of the latitude arrays. Defaults to "latitudes".
        lon_key : str, optional
            Name of the key of the latitude arrays. Defaults to "longitudes".
        """
        self.npz_file = Path(npz_file)
        self.lat_key = lat_key
        self.lon_key = lon_key
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        assert self.npz_file.exists(), f"{self.__class__.__name__}.file does not exists: {self.npz_file}"
        grid_data = np.load(self.npz_file)
        coords = self.reshape_coords(grid_data[self.lat_key], grid_data[self.lon_key])
        return coords


class LimitedAreaNPZFileNodes(NPZFileNodes):
    """Nodes from NPZ defined grids using an area of interest."""

    def __init__(
        self,
        npz_file: str,
        reference_node_name: str,
        name: str,
        lat_key: str = "latitudes",
        lon_key: str = "longiutdes",
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

        super().__init__(npz_file, name, lat_key, lon_key)

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


class XArrayNodes(BaseNodeBuilder):
    """Class for creating graph nodes based on a xarray-compatible file format.

    Parameters
    ----------
    dataset : str
        Path to xarray compatible file (e.g., NetCDF or zarr) containing latitude and longitude variables.
    name : str
        Identifier to use for the nodes within the graph.
    lat_key : str, optional
        Variable name for latitude in the dataset (default: "lat").
    lon_key : str, optional
        Variable name for longitude in the dataset (default: "lon").

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

    def __init__(self, dataset: str, name: str, lat_key: str = "lat", lon_key: str = "lon") -> None:

        super().__init__(name)
        self.dataset = dataset
        self.lat_key = lat_key
        self.lon_key = lon_key
        self.hidden_attributes = BaseNodeBuilder.hidden_attributes | {"dataset"}

    def get_coordinates(self) -> torch.Tensor:
        import xarray as xr

        ds = xr.open_dataset(self.dataset)

        for var in [self.lat_key, self.lon_key]:
            assert var in ds, f"Variable '{var}' not found in dataset."

        lat = ds[self.lat_key].values.flatten()
        lon = ds[self.lon_key].values.flatten()
        return self.reshape_coords(lat, lon)
