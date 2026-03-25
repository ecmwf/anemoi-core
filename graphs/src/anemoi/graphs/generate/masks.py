# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time
from importlib.util import find_spec

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS

LOGGER = logging.getLogger(__name__)

TORCH_CLUSTER_AVAILABLE = find_spec("torch_cluster") is not None


class RadiusAreaMaskBuilder:
    """Class to build a mask based on a radius search using the similarity of the Euclidean chord length between
    two points on the unit sphere and their great-circle distance.

    If torch_cluster is available: all calculations are done as torch tensors!
        * use_gpu = True: Calculations are done on the GPU
        * use_gpu = False: Calculations are done on the CPU (seems to be fasted from the benchmarks)
    Else falling back to scipy.spatial.cKDTree (on CPU)
    """

    def __init__(
        self,
        reference_node_name: str,
        margin_radius_km: float = 100,
        mask_attr_name: str | None = None,
        use_gpu: bool = False,
    ):
        """Initialisation of the RadiusMaskBuilder
        The use_gpu optional argument is introduced for testing, but is actually never set to True when used by anemoi-graphs' CLI
        """
        assert isinstance(margin_radius_km, (int, float)), "The margin radius must be a number."
        assert margin_radius_km > 0, "The margin radius must be positive."

        self.margin_radius_km = margin_radius_km
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

        self._ref_vectors: torch.Tensor | np.ndarray | None = None
        self._kdtree: cKDTree | None = None

        self.use_gpu = use_gpu
        if use_gpu and not torch.cuda.is_available():
            LOGGER.warning("No GPU available falling back to CPU")
            self.use_gpu = False
        if use_gpu and not TORCH_CLUSTER_AVAILABLE:
            LOGGER.warning("The 'torch-cluster library is not installed, cannot use the GPU. Falling back scipy + CPU")
            self.use_gpu = False

    @property
    def _chord_threshold(self) -> float:
        """Euclidean chord length threshold equivalent to margin_radius_km."""
        return float(2 * np.sin(self.margin_radius_km / (2 * EARTH_RADIUS)))

    @staticmethod
    def _to_unit_sphere_torch(coords_rad: torch.Tensor) -> torch.Tensor:
        """Convert (lat, lon) in radians to 3D unit-sphere coordinates as torch.Tensors.

        Parameters
        ----------
        coords_rad : torch.Tensor of shape (N, 2)
            Latitude and longitude in radians.

        Returns
        -------
        torch.Tensor of shape (N, 3)
            Unit-sphere Cartesian coordinates, in float64.
        """
        LOGGER.debug(f"coords_rad is on device: {coords_rad.device}")
        # TODO: We can cast to torch.float32 for larger chunksize (faster computations)
        # But then graphs old graphs are not 100% reproducable (typically have 3 ~ 5 more points at the mask boundary)
        coords_rad = coords_rad.to(torch.float64)
        lat, lon = coords_rad[:, 0], coords_rad[:, 1]
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=1)

    @staticmethod
    def _to_unit_sphere(coords_rad: np.ndarray) -> np.ndarray:
        """Convert (lat, lon) in radians to 3D unit-sphere coordinates as numpy.ndarrays.

        Parameters
        ----------
        coords_rad : numpy.ndarray of shape (N, 2)
            Latitude and longitude in radians.

        Returns
        -------
        numpy.ndarray of shape (N, 3)
            Unit-sphere Cartesian coordinates, in float64.
        """
        LOGGER.debug(f"coords_rad is on device: {coords_rad.device}")
        lat, lon = coords_rad[:, 0], coords_rad[:, 1]
        LOGGER.debug(f"lat.dtype = {lat.dtype}")
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.stack([x, y, z], axis=1)

    def get_reference_coords(self, graph: HeteroData) -> torch.Tensor:
        """Retrieve coordinates from the reference nodes (kept on device).

        Parameters
        ----------
        graph : HeteroData
            Graph object containing the reference nodes.

        Returns
        -------
        torch.Tensor of shape (N_ref, 2)
            Latitude and longitude of the reference nodes in radians.
        """
        assert (
            self.reference_node_name in graph.node_types
        ), f'Reference node "{self.reference_node_name}" not found in the graph.'

        coords_rad = graph[self.reference_node_name].x
        if self.mask_attr_name is not None:
            assert (
                self.mask_attr_name in graph[self.reference_node_name].node_attrs()
            ), f'Mask attribute "{self.mask_attr_name}" not found in the reference nodes.'
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]

        return coords_rad

    def fit_coords(self, coords_rad: torch.Tensor) -> None:
        """Store the reference unit-sphere vectors.

        Parameters
        ----------
        coords_rad : torch.Tensor of shape (N_ref, 2)
            Latitude and longitude of the reference nodes in radians.
        """
        if TORCH_CLUSTER_AVAILABLE:
            self._ref_vectors = self._to_unit_sphere_torch(coords_rad)
        else:
            self._ref_vectors = self._to_unit_sphere(coords_rad)
            self._kdtree = cKDTree(self._ref_vectors)

    def fit(self, graph: HeteroData) -> None:
        """Fit to the reference nodes in the graph.

        Parameters
        ----------
        graph : HeteroData
            Graph object containing the reference nodes.
        """
        reference_mask_str = self.reference_node_name
        if self.mask_attr_name is not None:
            reference_mask_str += f" ({self.mask_attr_name})"

        coords_rad = self.get_reference_coords(
            graph
        )  # This is always a torch.Tensor | when cluster is available coords_rad lives on the GPU
        if not TORCH_CLUSTER_AVAILABLE:
            self.fit_coords(coords_rad.cpu().numpy())
        elif not self.use_gpu:
            self.fit_coords(coords_rad.cpu())
        else:
            self.fit_coords(coords_rad)

        LOGGER.info(
            'Fitting %s with %d reference nodes from "%s".',
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )

    def get_mask(self, coords_rad: torch.Tensor) -> torch.Tensor:
        """Compute a mask based on the distance to the reference nodes.

        For each query node, checks whether it lies within margin_radius_km of
        any reference node, using the Euclidean chord distance equivalent to margin_radius_km as threshold.

        Parameters
        ----------
        coords_rad : torch.Tensor of shape (N_query, 2)
            Latitude and longitude of the query nodes in radians.

        Returns
        -------
        torch.Tensor of shape (N_query,)
            Boolean mask, True where the query node is within margin_radius_km
            of at least one reference node.

        """
        t0 = time.time()

        # The coords_rad from the query (typically the processor/hidden nodes) are produced on the CPU as numpy.ndarray.
        # Need to convert them to torch.Tensor if TORCH_CLUSER_AVAILABLE and move them to the correct device.
        if TORCH_CLUSTER_AVAILABLE:
            from torch_geometric.nn import radius

            LOGGER.debug("Using torch-cluster.radius")

            assert self._ref_vectors is not None, "The model must be fitted before calling get_mask."

            query_vectors = self._to_unit_sphere_torch(
                torch.from_numpy(coords_rad).to(self._ref_vectors.device)
            )  # (N_query, 3)
            LOGGER.debug(
                f"Reference vectors are on {self._ref_vectors.device}, query vectors are one {query_vectors.device}"
            )
            edge_index = radius(
                x=self._ref_vectors,  # reference points
                y=query_vectors,  # query points
                r=self._chord_threshold,
                max_num_neighbors=1,
            )

            mask = torch.zeros(len(query_vectors), dtype=torch.bool, device=self._ref_vectors.device)
            mask[edge_index[0]] = True
            # Bring the mask back to the CPU and return as numpy.ndarray
            mask = mask.cpu().numpy()
        else:
            LOGGER.debug("Using cKDTree")

            assert self._kdtree is not None, "The model must be fitted before calling get_mask."
            query_vectors = self._to_unit_sphere(coords_rad)
            LOGGER.debug("Reference vectors are on cpu, query vectors are one cpu")
            counts = self._kdtree.query_ball_point(
                query_vectors, r=self._chord_threshold, workers=-1, return_length=True
            )
            mask = counts > 0

        t1 = time.time()
        LOGGER.debug("Time to get mask from (%s): %.2f s", self.__class__.__name__, t1 - t0)

        return mask


class KNNAreaMaskBuilder:
    """Class to build a mask based on distance to masked reference nodes using KNN.

    Attributes
    ----------
    nearest_neighbour : NearestNeighbors
        Nearest neighbour object to compute the KNN.
    margin_radius_km : float
        Maximum distance to the reference nodes to consider a node as valid, in kilometers. Defaults to 100 km.
    reference_node_name : str
        Name of the reference nodes in the graph to consider for the Area Mask.
    mask_attr_name : str
        Name of a node to attribute to mask the reference nodes, if desired. Defaults to consider all reference nodes.

    Methods
    -------
    fit_coords(coords_rad: np.ndarray)
        Fit the KNN model to the coordinates in radians.
    fit(graph: HeteroData)
        Fit the KNN model to the reference nodes.
    get_mask(coords_rad: np.ndarray) -> np.ndarray
        Get the mask for the nodes based on the distance to the reference nodes.
    """

    def __init__(self, reference_node_name: str, margin_radius_km: float = 100, mask_attr_name: str | None = None):
        assert isinstance(margin_radius_km, (int, float)), "The margin radius must be a number."
        assert margin_radius_km > 0, "The margin radius must be positive."

        self.nearest_neighbour = NearestNeighbors(metric="haversine", n_jobs=4)
        self.margin_radius_km = margin_radius_km
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

    def get_reference_coords(self, graph: HeteroData) -> np.ndarray:
        """Retrive coordinates from the reference nodes."""
        assert (
            self.reference_node_name in graph.node_types
        ), f'Reference node "{self.reference_node_name}" not found in the graph.'

        coords_rad = graph[self.reference_node_name].x
        if self.mask_attr_name is not None:
            assert (
                self.mask_attr_name in graph[self.reference_node_name].node_attrs()
            ), f'Mask attribute "{self.mask_attr_name}" not found in the reference nodes.'
            mask = graph[self.reference_node_name][self.mask_attr_name].squeeze()
            coords_rad = coords_rad[mask]
        LOGGER.debug(f"The coords_rad tensor is on {coords_rad.device}")
        return coords_rad.cpu().numpy()

    def fit_coords(self, coords_rad: np.ndarray):
        """Fit the KNN model to the coordinates in radians."""
        self.nearest_neighbour.fit(coords_rad)

    def fit(self, graph: HeteroData):
        """Fit the KNN model to the nodes of interest."""
        # Prepare string for logging
        reference_mask_str = self.reference_node_name
        if self.mask_attr_name is not None:
            reference_mask_str += f" ({self.mask_attr_name})"

        # Fit to the reference nodes
        coords_rad = self.get_reference_coords(graph)
        self.fit_coords(coords_rad)

        LOGGER.info(
            'Fitting %s with %d reference nodes from "%s".',
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )

    def get_mask(self, coords_rad: np.ndarray) -> np.ndarray:
        """Compute a mask based on the distance to the reference nodes."""
        LOGGER.debug("Getting mask from KNNAreaMaskBuilder")
        t0 = time.time()
        neigh_dists, _ = self.nearest_neighbour.kneighbors(coords_rad, n_neighbors=1)
        mask = neigh_dists[:, 0] * EARTH_RADIUS <= self.margin_radius_km
        t1 = time.time()
        LOGGER.debug("Time to get mask from (%s): %.2f s", self.__class__.__name__, t1 - t0)

        return mask
