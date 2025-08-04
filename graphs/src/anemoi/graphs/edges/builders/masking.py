# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian

LOGGER = logging.getLogger(__name__)


class NodeMaskingMixin:
    """Mixin class for masking source/target nodes when building edges."""

    def get_node_coordinates(
        self, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the node coordinates."""
        source_coords, target_coords = source_nodes.x, target_nodes.x

        if self.source_mask_attr_name is not None:
            source_coords = source_coords[source_nodes[self.source_mask_attr_name].squeeze()]

        if self.target_mask_attr_name is not None:
            target_coords = target_coords[target_nodes[self.target_mask_attr_name].squeeze()]

        return source_coords.to(self.device), target_coords.to(self.device)

    def get_cartesian_node_coordinates(
        self, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cartesian coordinates of the source and target nodes.

        This method handles also the masking of source/target nodes if specified.

        Attributes
        ----------
        source_nodes : NodeStorage
            The source node storage
        target_nodes : NodeStorage
            The target node storage

        Return
        ------
        source_coords : torch.Tensor
            Source coordinates.
        target_coords : torch.Tensor
            Target coordinates.
        """
        source_coords, target_coords = self.get_node_coordinates(source_nodes, target_nodes)
        source_coords = latlon_rad_to_cartesian(source_coords)
        target_coords = latlon_rad_to_cartesian(target_coords)
        return source_coords, target_coords

    @staticmethod
    def get_unmasking_mapping(mask: torch.Tensor) -> Callable:
        masked_indices = np.where(mask.squeeze().cpu())[0]
        mapper = dict(zip(range(len(masked_indices)), masked_indices))
        return np.vectorize(mapper.get)

    def undo_masking_adj_matrix(self, adj_matrix, source_nodes: NodeStorage, target_nodes: NodeStorage):
        """Undo masking for adjacency matrix, remapping indices to original node indices.

        Arguments
        ---------
        adj_matrix
            Adjacency matrix
        source_nodes : NodeStorage
            Source node storage.
        target_nodes : NodeStorage
            Target node storage.

        Returns
        -------
        np.ndarray
            Remapped adj_matrix with original node indices.
        """
        if self.target_mask_attr_name is not None:
            target_node_mapping = NodeMaskingMixin.get_unmasking_mapping(mask=target_nodes[self.target_mask_attr_name])
            adj_matrix.row = target_node_mapping(adj_matrix.row)

        if self.source_mask_attr_name is not None:
            source_node_mapping = NodeMaskingMixin.get_unmasking_mapping(mask=source_nodes[self.source_mask_attr_name])
            adj_matrix.col = source_node_mapping(adj_matrix.col)

        if self.source_mask_attr_name is not None or self.target_mask_attr_name is not None:
            true_shape = target_nodes.x.shape[0], source_nodes.x.shape[0]
            adj_matrix = coo_matrix((adj_matrix.data, (adj_matrix.row, adj_matrix.col)), shape=true_shape)

        return adj_matrix

    def undo_masking_edge_index(
        self, edge_index: torch.Tensor, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> torch.Tensor:
        """Undo masking for edge_index matrix, remapping indices to original node indices.

        Arguments
        ---------
        edge_index : torch.Tensor
            2 x N tensor of edge indices (target, source).
        source_nodes : NodeStorage
            Source node storage.
        target_nodes : NodeStorage
            Target node storage.

        Returns
        -------
        torch.Tensor
            Remapped edge_index with original node indices.
        """
        # Remap source indices (row 0)
        if self.source_mask_attr_name is not None:
            source_node_mapping = NodeMaskingMixin.get_unmasking_mapping(mask=source_nodes[self.source_mask_attr_name])
            edge_index[0] = torch.from_numpy(source_node_mapping(edge_index[0].cpu().numpy())).to(edge_index.device)

        # Remap target indices (row 1)
        if self.target_mask_attr_name is not None:
            target_node_mapping = NodeMaskingMixin.get_unmasking_mapping(mask=target_nodes[self.target_mask_attr_name])
            edge_index[1] = torch.from_numpy(target_node_mapping(edge_index[1].cpu().numpy())).to(edge_index.device)

        return edge_index
