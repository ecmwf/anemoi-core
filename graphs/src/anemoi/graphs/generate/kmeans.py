# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np
from sklearn.cluster import KMeans
from anemoi.graphs.utils import current_device_context
from anemoi.graphs.utils import get_distributed_device
from anemoi.graphs.utils import is_pyg_lib_available

LOGGER = logging.getLogger(__name__)


class KMeansClusterBuilder:
    """K-means cluster builder for reference nodes on a unit sphere.


    Methods
    -------
    fit_coords(coords_rad: torch.Tensor)
        Fit the backend to the coordinates in radians.
    fit(graph: HeteroData)
        Fit the backend to the reference nodes.
    get_mask(coords_rad: torch.Tensor | np.ndarray) -> torch.Tensor
        Get the mask for the nodes based on the distance to the reference nodes.
    """

    def __init__(
        self,
        reference_node_name: str,
        mask_attr_name: str | None = None,
        node_weight_attribute_name: str | None = None,
    ):
        """Initialisation of the KMeansClusterMaskBuilder."""
        self.reference_node_name = reference_node_name
        self.mask_attr_name = mask_attr_name
        self.node_weight_attribute_name = node_weight_attribute_name

        self.device = get_distributed_device()

        self._backend = None
        self._coords_xyz = None
        self._weights = None

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

        if self.node_weight_attribute_name is not None:
            assert (
                self.node_weight_attribute_name in graph[self.reference_node_name].node_attrs()
            ), f'Node weight attribute "{self.node_weight_attribute_name}" not found in the reference nodes.'
            weights = graph[self.reference_node_name][self.node_weight_attribute_name].squeeze()
            if self.mask_attr_name is not None:
                weights = weights[mask]
        else:
            weights = None

        return coords_rad, weights

    def fit(self, graph: HeteroData) -> None:
        """Fit the K-means model to the reference nodes in the graph."""
        reference_mask_str = self.reference_node_name
        if self.mask_attr_name is not None:
            reference_mask_str += f"[{self.mask_attr_name}]"

        coords_rad, weights = self._get_reference_coords(graph)
        LOGGER.info(
            "Fitting %s (%s) with %d reference nodes from %s.",
            self.__class__.__name__,
            self._backend.__class__.__name__,
            len(coords_rad),
            reference_mask_str,
        )
        LOGGER.debug("Reference nodes live on %s", coords_rad.device)

        self._coords_xyz = latlon_rad_to_cartesian_np(coords_rad)
        self._weights = weights

    def get_clusters(self, ratio: float, **kwargs) -> np.ndarray:
        """Get the cluster centers in latitude and longitude (radians)."""
        if self._coords_xyz is None:
            raise RuntimeError("No coordinates available. Fit the model first.")

        num_points = len(self._coords_xyz)
        num_clusters = max(1, round(num_points / ratio))
        self._backend = KMeans(n_clusters=num_clusters, **kwargs)
        LOGGER.info(
            "Fitting K-means with %d clusters for %d reference points.",
            num_clusters,
            num_points,
        )

        self._backend.fit(self._coords_xyz, sample_weight=self._weights)
        return cartesian_to_latlon_rad(self._backend.cluster_centers_)
