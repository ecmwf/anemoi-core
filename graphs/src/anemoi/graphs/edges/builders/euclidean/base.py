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

from torch_geometric.data.storage import NodeStorage
from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from anemoi.graphs.edges.builders.masking import NodeMaskingMixin
from abc import ABC
from abc import abstractmethod
from anemoi.graphs.utils import cuda_device_of
from anemoi.graphs.utils import is_pyg_lib_available


LOGGER = logging.getLogger(__name__)


PYG_LIB_INSTRUCTIONS = r"""The 'pyg_lib' library is not installed.
Installing 'pyg_lib' can significantly improve performance for graph creation.
You can install it using:
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
"""


class BaseDistanceEdgeBuilders(BaseEdgeBuilder, NodeMaskingMixin, ABC):
    """Base class for edge builders based on distance."""

    def prepare_method_kwargs(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> dict:
        """Prepare keyword arguments."""
        return {}

    @abstractmethod
    def _compute_edge_index_pyg(self, source_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _compute_adj_matrix_sklearn(self, source_coords: torch.Tensor, target_coords: torch.Tensor, **kwargs) -> np.ndarray: ...

    def compute_edge_index_from_coords(self, source_coords: torch.Tensor, target_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute edge index using pyg-lib (if available) or sklearn.

        Parameters
        ----------
        source_coords : torch.Tensor
            Coordinates of source nodes of shape (num_source_nodes, 3) in unit sphere.
        target_coords : torch.Tensor
            Coordinates of target nodes of shape (num_target_nodes, 3) in unit sphere.

        Returns
        -------
        torch.Tensor
            Edge index tensor of shape (2, num_edges).
        """
        # for an empty node set (an observation window with no points in this batch)
        # there are no edges, return an empty edge index
        if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
            return torch.empty((2, 0), dtype=torch.long, device=source_coords.device)

        # guard against empty node sets (an obs dataset window with no points in this batch)
        # short-circuit to an empty (2, 0) edge index
        if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
            return torch.empty((2, 0), dtype=torch.long, device=source_coords.device)

        if is_pyg_lib_available():
            # pyg-lib's kernels install no device guard of their own; see cuda_device_of.
            with cuda_device_of(source_coords.device):
                edge_index = self._compute_edge_index_pyg(source_coords, target_coords, **kwargs)
        else:
            LOGGER.warning(PYG_LIB_INSTRUCTIONS)
            adj_matrix = self._compute_adj_matrix_sklearn(source_coords, target_coords, **kwargs)
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
        source_coords, target_coords = self.get_cartesian_node_coordinates(source_nodes, target_nodes) # 3d coords
        method_kwargs = self.prepare_method_kwargs(source_coords, target_coords)
        edge_index = self.compute_edge_index_from_coords(source_coords, target_coords, **method_kwargs)
        edge_index = self.undo_masking_edge_index(edge_index, source_nodes, target_nodes)
        return edge_index
