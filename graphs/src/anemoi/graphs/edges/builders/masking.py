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

import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data.storage import NodeStorage

LOGGER = logging.getLogger(__name__)


class NodeMaskingMixin:
    """Mixin class for masking source/target nodes when building edges."""

    def get_node_coordinates(
        self, source_nodes: NodeStorage, target_nodes: NodeStorage
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the node coordinates."""
        source_coords, target_coords = source_nodes.x.numpy(), target_nodes.x.numpy()

        if self.source_mask_attr_name is not None:
            source_coords = source_coords[source_nodes[self.source_mask_attr_name].squeeze()]

        if self.target_mask_attr_name is not None:
            target_coords = target_coords[target_nodes[self.target_mask_attr_name].squeeze()]

        return source_coords, target_coords

    def undo_masking(self, adj_matrix, source_nodes: NodeStorage, target_nodes: NodeStorage):
        if self.target_mask_attr_name is not None:
            target_mask = target_nodes[self.target_mask_attr_name].squeeze()
            assert adj_matrix.shape[0] == target_mask.sum()
            target_mapper = dict(zip(list(range(adj_matrix.shape[0])), np.where(target_mask)[0]))
            adj_matrix.row = np.vectorize(target_mapper.get)(adj_matrix.row)

        if self.source_mask_attr_name is not None:
            source_mask = source_nodes[self.source_mask_attr_name].squeeze()
            assert adj_matrix.shape[1] == source_mask.sum()
            source_mapper = dict(zip(list(range(adj_matrix.shape[1])), np.where(source_mask)[0]))
            adj_matrix.col = np.vectorize(source_mapper.get)(adj_matrix.col)

        if self.source_mask_attr_name is not None or self.target_mask_attr_name is not None:
            true_shape = target_nodes.x.shape[0], source_nodes.x.shape[0]
            adj_matrix = coo_matrix((adj_matrix.data, (adj_matrix.row, adj_matrix.col)), shape=true_shape)

        return adj_matrix
