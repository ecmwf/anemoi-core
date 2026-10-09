# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.edges.builders.masking import NodeMaskingMixin
from anemoi.graphs.utils import PYG_INSTRUCTIONS
from anemoi.graphs.utils import cuda_device_of
from anemoi.graphs.utils import is_pyg_lib_available

LOGGER = logging.getLogger(__name__)


class BaseDistanceEdgeBuilders(BaseEdgeBuilder, NodeMaskingMixin, ABC):
    """Base class for edge builders based on distance."""

    #: Whether the neighbour search runs from the source nodes rather than from the target nodes, as
    #: the ``Reversed*`` builders do. The coordinates are swapped before the search and the result is
    #: left unflipped, so the edge index still comes back with (source, target) rows.
    reversed_search: bool = False

    def prepare_method_kwargs(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> dict:
        """Prepare keyword arguments."""
        return {}

    @abstractmethod
    def _compute_edge_index_pyg(
        self, source_coords: torch.Tensor, target_coords: torch.Tensor, skip_flip: bool = False, **kwargs
    ) -> torch.Tensor: ...

    @abstractmethod
    def _compute_adj_matrix_sklearn(
        self, source_coords: torch.Tensor, target_coords: torch.Tensor, **kwargs
    ) -> np.ndarray: ...

    def compute_edge_index_from_coords(
        self, source_coords: torch.Tensor, target_coords: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute edge index using pyg-lib (if available) or sklearn.

        This is the entry point for both :meth:`compute_edge_index` at graph-build time and
        ``anemoi.models.layers.graph_provider.DynamicGraphProvider`` at runtime, which passes the
        per-batch coordinates and no keyword arguments.

        Parameters
        ----------
        source_coords : torch.Tensor
            Coordinates of source nodes of shape (num_source_nodes, 3) in unit sphere.
        target_coords : torch.Tensor
            Coordinates of target nodes of shape (num_target_nodes, 3) in unit sphere.
        **kwargs
            Keyword arguments for the backends, overriding those from :meth:`prepare_method_kwargs`.

        Returns
        -------
        torch.Tensor
            Edge index tensor of shape (2, num_edges).
        """
        # guard against empty node sets (an obs dataset window with no points in this batch)
        # short-circuit to an empty (2, 0) edge index
        if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
            return torch.empty((2, 0), dtype=torch.long, device=source_coords.device)

        # Resolve before the swap below: prepare_method_kwargs must see the real source/target, as
        # ReversedCutOffEdges derives its radius from the source nodes.
        kwargs = self.prepare_method_kwargs(source_coords, target_coords) | kwargs

        skip_flip = self.reversed_search
        if skip_flip:
            source_coords, target_coords = target_coords, source_coords

        if is_pyg_lib_available():
            # pyg-lib's kernels install no device guard of their own; see cuda_device_of.
            with cuda_device_of(source_coords.device):
                edge_index = self._compute_edge_index_pyg(source_coords, target_coords, skip_flip=skip_flip, **kwargs)
        else:
            LOGGER.warning(PYG_INSTRUCTIONS)
            adj_matrix = self._compute_adj_matrix_sklearn(source_coords, target_coords, **kwargs)

            if skip_flip:
                edge_index = torch.from_numpy(np.stack([adj_matrix.row, adj_matrix.col], axis=0))
            else:
                edge_index = torch.from_numpy(np.stack([adj_matrix.col, adj_matrix.row], axis=0))

        return edge_index

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge indices.

        Parameters
        ----------
        source_nodes : NodeStorage
            The source nodes.
        target_nodes : NodeStorage
            The target nodes.

        Returns
        -------
        torch.Tensor of shape (2, num_edges)
            Indices of source and target nodes connected by an edge.
        """
        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes)  # 3d coords
        edge_index = self.compute_edge_index_from_coords(source_coords, target_coords)
        edge_index = self.undo_masking_edge_index(edge_index, source_nodes, target_nodes)
        return edge_index
