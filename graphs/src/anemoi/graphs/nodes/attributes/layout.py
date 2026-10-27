# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class RectilinearLayoutIndex(BaseNodeAttribute):
    """Row and column of each node in its rectilinear grid.

    It is defined for nodes built with ``RectilinearNodes`` or ``LimitedAreaRectilinearNodes``.
    The grid size is in the hidden node attribute ``_layout_shape``, so a field on the nodes
    can be put back on the full grid, with NaN at the grid points that are not nodes:

    .. code:: python

        index = graph["hidden"]["layout_index"]
        grid = torch.full(graph["hidden"]["_layout_shape"], float("nan"))
        grid[index[:, 0], index[:, 1]] = values

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self) -> None:
        super().__init__(norm=None, dtype="int64")

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert "_layout_index" in nodes, (
            f"{self.__class__.__name__} can only be used with nodes built with RectilinearNodes "
            f"or LimitedAreaRectilinearNodes, got {nodes['node_type']}."
        )
        return nodes["_layout_index"]
