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
    """Base class for creating node attributes.

    It uses the information provided in the config to describe how the attribute is computed.

    It expects the following attributes to be defined:
    - `name` defines the name of the node attribute being created. This will be used to store the computed node
    attributes in the graph data structure (`:class:torch_geometric.data.HeteroData`). It can be set either as a
    class attribute or in `__init__`. It is only required when the attribute is registered in the graph, so
    attributes used as building blocks of other attributes (e.g. the masks of a boolean operation) can omit it.

    There are other arguments that can be specified too:
    - `norm_by_group` specifies whether the node attribute should be normalized by group. The default is False.
    - `dtype` specifies the data type of the node attribute. The default is "float32".
    - `norm` specifies the normalization method to be applied to the node attribute. The default is None, meaning
    no normalization is applied. Options are:
        - "unit-range": scales the node attribute to the range [0, 1].
        - "unit-std": standardizes the node attribute to have zero mean and unit variance.
        - "unit-max": scales the node attribute by its maximum value.
        - "l1": normalizes the node attribute using the L1 norm.
        - "l2": normalizes the node attribute using the L2 norm.
        - None: no normalization is applied.

    The `_get_raw_values` method must be implemented by subclasses to define how the node attribute
    is computed from the node storage object. The coordinates are stored in the `x` attribute of the node storage,
    and any previously registered node attributes can be accessed too.

    Methods
    -------
    compute(graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor
        Computes the node attribute for the nodes `nodes_name` of the graph.

    Example
    -------
        class LatitudeAttribute(BaseNodeAttribute):
            name = "latitude"

            def _get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
                return nodes.x[:, 0]
    """

    name: str | None = None
    norm_by_group: bool = False

    def __init__(self, name: str | None = None, norm: str | None = None, dtype: str = "float32") -> None:
        self.name = name
        self.norm = norm
        self.dtype = getattr(torch, dtype)
        self.device = get_distributed_device()

    @abstractmethod
    def _get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        """Compute the raw (unnormalised) values of the node attribute.

        Parameters
        ----------
        nodes : NodeStorage
            Nodes of the graph.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor (num_nodes,) or (num_nodes, num_node_features)
            Raw values of the node attribute.
        """

    def _post_process(self, values: torch.Tensor) -> torch.Tensor:
        """Post-process the values.

        It ensures the values are 2D, i.e. (num_nodes, num_node_features), and normalises them.

        Parameters
        ----------
        values : torch.Tensor
            Raw values of the node attribute.

        Returns
        -------
        torch.Tensor (num_nodes, num_node_features)
            Post-processed values of the node attribute.
        """
        if values.ndim == 1:
            values = torch.unsqueeze(values, -1)

        return self.normalise(values)

    def compute(self, graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor:
        """Compute the node attribute for the given nodes of the graph.

        Parameters
        ----------
        graph : HeteroData
            Graph containing the nodes.
        nodes_name : str
            Name of the nodes for which to compute the attribute.
        kwargs : dict
            Additional keyword arguments, passed to `_get_raw_values`.

        Returns
        -------
        torch.Tensor (num_nodes, num_node_features)
            The computed node attributes.
        """
        assert (
            nodes_name in graph.node_types
        ), f"{nodes_name} is not a valid nodes name. The current graph has the following nodes: {graph.node_types}"
        nodes = graph[nodes_name].to(self.device)
        attributes = self._get_raw_values(nodes, **kwargs).to(dtype=self.dtype, device=self.device)
        return self._post_process(attributes)


class BooleanBaseNodeAttribute(BaseNodeAttribute, ABC):
    """Base class for boolean node attributes.

    Boolean attributes (e.g. masks) are never normalised and are stored with dtype "bool".

    Attributes
    ----------
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name, norm=None, dtype="bool")
