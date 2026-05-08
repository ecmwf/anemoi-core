# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional

import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.edges.builders.base import BaseEdgeBuilder

LOGGER = logging.getLogger(__name__)


class HEALPixMultiScaleEdges(BaseEdgeBuilder):
    """HEALPix Multi Scale Edges."""

    def __init__(
        self,
        source_name: str,
        target_name: str,
        scale_resolutions: Optional[int | list[int]] = None,
        **kwargs,
    ):
        super().__init__(source_name, target_name)
        assert source_name == target_name, f"{self.__class__.__name__} requires source and target nodes to be the same."
        if isinstance(scale_resolutions, int):
            assert scale_resolutions > 0, "The scale_resolutions argument only supports positive integers."
            scale_resolutions = list(range(1, scale_resolutions + 1))
        assert not isinstance(scale_resolutions, str), "The scale_resolutions argument is not valid."
        assert (
            scale_resolutions is None or min(scale_resolutions) > 0
        ), "The scale_resolutions argument only supports positive integers."
        self.scale_resolutions = scale_resolutions

    def compute_edge_index(self, source_nodes: NodeStorage, target_nodes: NodeStorage) -> torch.Tensor:
        """Compute the edge index for HEALPix multi scale edges."""
        assert source_nodes.node_type == "HEALPixNodes", f"{self.__class__.__name__} only supports HEALPixNodes."

        from anemoi.graphs.generate.healpix import get_healpix_edgeindex

        scale_resolutions = self.scale_resolutions or list(range(1, source_nodes["_resolution"] + 1))
        edges_index, prev_res = None, None
        for res in list(sorted(scale_resolutions)):
            new_edge_index = get_healpix_edgeindex(res)
            LOGGER.debug(f"Resolution: {res}, Edge index shape: {new_edge_index.shape}")
            if edges_index is None:
                edges_index = new_edge_index
            else:
                edges_index = torch.cat([4 ** (res - prev_res) * edges_index, new_edge_index], dim=1)
            prev_res = res

        return edges_index
