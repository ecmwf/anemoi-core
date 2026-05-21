# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from importlib.util import find_spec

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np
from anemoi.graphs.utils import get_distributed_device

LOGGER = logging.getLogger(__name__)

TORCH_CLUSTER_AVAILABLE = find_spec("torch_cluster") is not None


class _TorchClusterAreaMaskBackend:
    """Torch-cluster radius backend (CPU/GPU depending on distributed device)."""

    def __init__(self, device: torch.device | str):
        self.device = device
        self._ref_vectors: torch.Tensor | None = None
        self._kdtree: cKDTree | None = None

    def fit(self, coords_rad: torch.Tensor) -> None:
        self._ref_vectors = latlon_rad_to_cartesian(coords_rad.to(self.device))

    def get_mask(self, coords_rad: torch.Tensor | np.ndarray, chord_threshold: float) -> torch.Tensor:
        from torch_geometric.nn import radius

        assert self._ref_vectors is not None, "The model must be fitted before calling get_mask."

        if isinstance(coords_rad, np.ndarray):
            coords_rad = torch.from_numpy(coords_rad)

        coords_rad = coords_rad.to(self.device)

        query_vectors = latlon_rad_to_cartesian(coords_rad)
        LOGGER.debug(
            "Reference vectors are on %s, query vectors are on %s",
            self._ref_vectors.device,
            query_vectors.device,
        )
        edge_index = radius(
            x=self._ref_vectors,
            y=query_vectors,
            r=chord_threshold,
            max_num_neighbors=1,
        )

        mask = torch.zeros(len(query_vectors), dtype=torch.bool, device=self._ref_vectors.device)
        mask[edge_index[0]] = True
        return mask.cpu()


class _KDTreeAreaMaskBackend:
    """Scipy cKDTree backend running on CPU."""

    def __init__(self):
        self._ref_vectors: np.ndarray | None = None
        self._kdtree: cKDTree | None = None

    def fit(self, coords_rad: torch.Tensor) -> None:
        self._ref_vectors = latlon_rad_to_cartesian_np(coords_rad.cpu().numpy())
        self._kdtree = cKDTree(self._ref_vectors)

    def get_mask(self, coords_rad: torch.Tensor | np.ndarray, chord_threshold: float) -> torch.Tensor:
        assert self._kdtree is not None, "The model must be fitted before calling get_mask."

        if isinstance(coords_rad, torch.Tensor):
            coords_rad = coords_rad.cpu().numpy()

        query_vectors = latlon_rad_to_cartesian_np(coords_rad)
        LOGGER.debug("Reference vectors are on cpu, query vectors are on cpu")
        counts = self._kdtree.query_ball_point(query_vectors, r=chord_threshold, workers=-1, return_length=True)
        return torch.from_numpy(counts > 0)


class AreaMaskBuilder:
    """Area mask builder using radius queries on unit-sphere chord distances.

    The public API is backend-agnostic. At runtime, a dedicated backend is selected:
    - torch-cluster backend when available
    - scipy cKDTree backend otherwise

    Methods
    -------
    fit_coords(coords_rad: torch.Tensor)
        Fit the KNN model to the coordinates in radians.
    fit(graph: HeteroData)
        Fit the KNN model to the reference nodes.
    get_mask(coords_rad: np.ndarray) -> np.ndarray
        Get the mask for the nodes based on the distance to the reference nodes.
    """

    def __init__(
        self,
        reference_node_name: str,
        margin_radius_km: float = 100,
        mask_attr_name: str | None = None,
    ):
        """Initialisation of the AreaMaskBuilder."""
        assert isinstance(margin_radius_km, (int, float)), "The margin radius must be a number."
        assert margin_radius_km > 0, "The margin radius must be positive."

        self.margin_radius_km = margin_radius_km
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name

        self.device = get_distributed_device()
        if TORCH_CLUSTER_AVAILABLE:
            self._backend = _TorchClusterAreaMaskBackend(device=self.device)
        else:
            self._backend = _KDTreeAreaMaskBackend()

    @property
    def _chord_threshold(self) -> float:
        """Euclidean chord length threshold equivalent to margin_radius_km."""
        return float(2 * np.sin(self.margin_radius_km / (2 * EARTH_RADIUS)))

    def _get_reference_coords(self, graph: HeteroData) -> torch.Tensor:
        """Retrieve coordinates from the reference nodes.

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
        LOGGER.debug(
            "%s: Using backend %s to fit coordinates with chord threshold %.4f",
            self.__class__.__name__,
            self._backend.__class__.__name__,
            self._chord_threshold,
        )

        self._backend.fit(coords_rad)

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

        coords_rad = self._get_reference_coords(graph)
        self.fit_coords(coords_rad)

        LOGGER.info(
            'Fitting %s with %d reference nodes from "%s".',
            self.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )

    def get_mask(self, coords_rad: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute a mask based on the distance to the reference nodes.

        For each query node, checks whether it lies within margin_radius_km of
        any reference node, using the Euclidean chord distance equivalent to margin_radius_km as threshold.

        Parameters
        ----------
        coords_rad : torch.Tensor or np.ndarray of shape (N_query, 2)
            Latitude and longitude of the query nodes in radians.

        Returns
        -------
        torch.Tensor of shape (N_query,)
            Boolean mask, True where the query node is within margin_radius_km
            of at least one reference node.
        """
        LOGGER.debug(
            "%s: Using backend %s to compute mask with chord threshold %.4f",
            self.__class__.__name__,
            self._backend.__class__.__name__,
            self._chord_threshold,
        )

        return self._backend.get_mask(coords_rad, chord_threshold=self._chord_threshold)
