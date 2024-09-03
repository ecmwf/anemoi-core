from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch_geometric.data import HeteroData

from anemoi.graphs.generate.hexagonal import create_hexagonal_nodes
from anemoi.graphs.generate.icosahedral import create_icosahedral_nodes

LOGGER = logging.getLogger(__name__)


class BaseNodeBuilder(ABC):
    """Base class for node builders.

    The node coordinates are stored in the `x` attribute of the nodes and they are stored in radians.

    Attributes
    ----------
    name : str
        name of the nodes, key for the nodes in the HeteroData graph object.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def register_nodes(self, graph: HeteroData) -> None:
        """Register nodes in the graph.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the nodes.
        """
        graph[self.name].x = self.get_coordinates()
        graph[self.name].node_type = type(self).__name__
        return graph

    def register_attributes(self, graph: HeteroData, config: DotDict = None) -> HeteroData:
        """Register attributes in the nodes of the graph specified.

        Parameters
        ----------
        graph : HeteroData
            The graph to register the attributes.
        config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with the registered attributes.
        """
        for attr_name, attr_config in config.items():
            graph[self.name][attr_name] = instantiate(attr_config).compute(graph, self.name)
        return graph

    @abstractmethod
    def get_coordinates(self) -> torch.Tensor: ...

    def reshape_coords(self, latitudes: np.ndarray, longitudes: np.ndarray) -> torch.Tensor:
        """Reshape latitude and longitude coordinates.

        Parameters
        ----------
        latitudes : np.ndarray of shape (N, )
            Latitude coordinates, in degrees.
        longitudes : np.ndarray of shape (N, )
            Longitude coordinates, in degrees.

        Returns
        -------
        torch.Tensor of shape (N, 2)
            A 2D tensor with the coordinates, in radians.
        """
        coords = np.stack([latitudes, longitudes], axis=-1).reshape((-1, 2))
        coords = np.deg2rad(coords)
        return torch.tensor(coords, dtype=torch.float32)

    def update_graph(self, graph: HeteroData, attr_config: DotDict | None = None) -> HeteroData:
        """Update the graph with new nodes.

        Parameters
        ----------
        graph : HeteroData
            Input graph.
        attr_config : DotDict
            The configuration of the attributes.

        Returns
        -------
        HeteroData
            The graph with new nodes included.
        """
        graph = self.register_nodes(graph)

        if attr_config is None:
            return graph

        graph = self.register_attributes(graph, attr_config)

        return graph


class ZarrDatasetNodes(BaseNodeBuilder):
    """Nodes from Zarr dataset.

    Attributes
    ----------
    dataset : zarr.core.Array
        The dataset.

    Methods
    -------
    register_nodes(graph)
        Register the nodes in the graph.
    register_attributes(graph, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, attr_config)
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
        torch.Tensor of shape (N, 2)
            Coordinates of the nodes.
        """
        return self.reshape_coords(self.dataset.latitudes, self.dataset.longitudes)


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
    register_nodes(graph)
        Register the nodes in the graph.
    register_attributes(graph, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, attr_config)
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
        torch.Tensor of shape (N, 2)
            Coordinates of the nodes.
        """
        coords = self.reshape_coords(self.grid_definition["latitudes"], self.grid_definition["longitudes"])
        return coords


class IcosahedralNodes(BaseNodeBuilder, ABC):
    """Processor mesh based on a triangular mesh.

    It is based on the icosahedral mesh, which is a mesh of triangles that covers the sphere.

    Parameters
    ----------
    resolution : list[int] | int
        Refinement level of the mesh.
    """

    def __init__(
        self,
        resolution: int | list[int],
        name: str,
    ) -> None:
        self.resolutions = list(range(resolution + 1)) if isinstance(resolution, int) else resolution
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        self.nx_graph, coords_rad, self.node_ordering = self.create_nodes()
        return torch.tensor(coords_rad[self.node_ordering], dtype=torch.float32)

    @abstractmethod
    def create_nodes(self) -> np.ndarray: ...

    def register_attributes(self, graph: HeteroData, config: DotDict) -> HeteroData:
        graph[self.name]["_resolutions"] = self.resolutions
        graph[self.name]["_nx_graph"] = self.nx_graph
        graph[self.name]["_node_ordering"] = self.node_ordering
        return super().register_attributes(graph, config)


class TriNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the trimesh Python library.

    Attributes
    ----------
    resolutions : list[int]
        Refinement level of the mesh.
    name : str
        The name of the nodes.

    Methods
    -------
    register_nodes(graph)
        Register the nodes in the graph.
    register_attributes(graph, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, attr_config)
        Update the graph with new nodes and attributes.
    """

    def create_nodes(self) -> np.ndarray:
        return create_icosahedral_nodes(resolutions=self.resolutions)


class HexNodes(IcosahedralNodes):
    """Nodes based on iterative refinements of an icosahedron.

    It depends on the h3 Python library.

    Attributes
    ----------
    resolutions : list[int]
        Refinement level of the mesh.
    name : str
        The name of the nodes.

    Methods
    -------
    register_nodes(graph)
        Register the nodes in the graph.
    register_attributes(graph, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, attr_config)
        Update the graph with new nodes and attributes.
    """

    def create_nodes(self) -> np.ndarray:
        return create_hexagonal_nodes(self.resolutions)


class HEALPixNodes(BaseNodeBuilder):
    """Nodes from HEALPix grid.

    HEALPix is an acronym for Hierarchical Equal Area isoLatitude Pixelization of a sphere.

    Attributes
    ----------
    resolution : int
        The resolution of the grid.
    name : str
        The name of the nodes.

    Methods
    -------
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attr_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, resolution: int, name: str) -> None:
        """Initialize the HEALPixNodes builder."""
        self.resolution = resolution
        super().__init__(name)

        assert isinstance(resolution, int), "Resolution must be an integer."
        assert resolution > 0, "Resolution must be positive."

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (N, 2)
            Coordinates of the nodes.
        """
        import healpy as hp

        spatial_res_degrees = hp.nside2resol(2**self.resolution, arcmin=True) / 60
        LOGGER.info(f"Creating HEALPix nodes with resolution {spatial_res_degrees:.2} deg.")

        npix = hp.nside2npix(2**self.resolution)
        hpxlon, hpxlat = hp.pix2ang(2**self.resolution, range(npix), nest=True, lonlat=True)

        return self.reshape_coords(hpxlat, hpxlon)
