# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import torch
from anemoi.graphs.edges.builders.base import BaseEdgeBuilder
from torch_geometric.data.storage import NodeStorage

LOGGER = logging.getLogger(__name__)


class HEALPixNNEdges(BaseEdgeBuilder):
    """HEALPix Nearest Neighbour Edges."""

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge index for HEALPix nearest neighbour edges."""
        from anemoi.graphs.generate.healpix import get_healpix_edgeindex

        resolution  = source_nodes["_resolution"]
        edges_index, prev_res = None, None
        for res in range(1, resolution + 1):
            new_edge_index = get_healpix_edgeindex(res)
            LOGGER.debug(f"Resolution: {res}, Edge index shape: {new_edge_index.shape}")
            if edges_index is None:
                edges_index = new_edge_index
            else:  
                edges_index = torch.cat([4 ** (res - prev_res) * edges_index, new_edge_index], dim=1)
            prev_res = res

        return edges_index
