# (C) Copyright 2024- Anemoi contributors.
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

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.normalise import NormaliserMixin
from anemoi.graphs.utils import get_distributed_device

LOGGER = logging.getLogger(__name__)


class BaseNodeAttribute(ABC, NormaliserMixin):
    """Base class for the weights of the nodes."""

    norm_by_group: bool = False

    def __init__(self, norm: str | None = None, dtype: str = "float32") -> None:
        self.norm = norm
        self.dtype = getattr(torch, dtype)
        self.device = get_distributed_device()

    @abstractmethod
    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor: ...

    def post_process(self, values: torch.Tensor) -> torch.Tensor:
        """Post-process the values."""
        if values.ndim == 1:
            values = torch.unsqueeze(values, -1)

        return self.normalise(values)

    def compute(self, graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor:
        """Get the nodes attribute.

        Parameters
        ----------
        graph : HeteroData
            Graph.
        nodes_name : str
            Name of the nodes.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Attributes associated to the nodes.
        """
        assert (
            nodes_name in graph.node_types
        ), f"{nodes_name} is not a valid nodes name. The current graph has the following nodes: {graph.node_types}"
        nodes = graph[nodes_name].to(self.device)
        attributes = self.get_raw_values(nodes, **kwargs).to(dtype=self.dtype, device=self.device)
        return self.post_process(attributes)


class BooleanBaseNodeAttribute(BaseNodeAttribute, ABC):
    """Base class for boolean node attributes."""

    def __init__(self) -> None:
        super().__init__(norm=None, dtype="bool")
