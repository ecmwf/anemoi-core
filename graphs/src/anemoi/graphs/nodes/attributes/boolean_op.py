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
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.nodes.attributes.base_attributes import BooleanBaseNodeAttribute

LOGGER = logging.getLogger(__name__)
MaskAttributeType = str | type["BooleanBaseNodeAttribute"]


class BooleanOperation(BooleanBaseNodeAttribute, ABC):
    """Base class for boolean operations over node masks.

    The `_reduce_op` method must be implemented by subclasses to define how the masks are combined.

    Attributes
    ----------
    masks : str | BooleanBaseNodeAttribute | list[str | BooleanBaseNodeAttribute]
        Masks to combine. Each mask is either the name of a boolean attribute already registered in the nodes,
        or a boolean node attribute builder, which is computed on the fly.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    _reduce_op(masks)
        Combine the stacked masks, of shape (num_masks, num_nodes), into a single mask.
    """

    def __init__(self, masks: MaskAttributeType | list[MaskAttributeType], name: str | None = None) -> None:
        super().__init__(name)
        assert masks is not None, f"{self.__class__.__name__} requires a valid masks argument."
        self.masks = masks if isinstance(masks, list) else [masks]

    @staticmethod
    def get_mask_values(mask: MaskAttributeType, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        if isinstance(mask, str):
            assert mask in nodes, f"Nodes have no attribute named {mask}."
            attributes = nodes[mask]
            assert (
                attributes.dtype == torch.bool
            ), f"The mask attribute '{mask}' must be a boolean but is {attributes.dtype}."
            return attributes

        return mask._get_raw_values(nodes, **kwargs)

    @abstractmethod
    def _reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor: ...

    def _get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        mask_values = [BooleanOperation.get_mask_values(mask, nodes, **kwargs) for mask in self.masks]
        return self._reduce_op(torch.stack(mask_values))


class BooleanNot(BooleanOperation):
    """Boolean NOT mask.

    It negates a single mask.

    Attributes
    ----------
    masks : str | BooleanBaseNodeAttribute | list[str | BooleanBaseNodeAttribute]
        Masks to combine. Each mask is either the name of a boolean attribute already registered in the nodes,
        or a boolean node attribute builder. Only one mask is allowed.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor
        Computes the NOT mask for the nodes `nodes_name` of the graph.
    """

    def _reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor:
        assert len(self.masks) == 1, f"The {self.__class__.__name__} can only be aplied to one mask."
        return ~masks[0]


class BooleanAndMask(BooleanOperation):
    """Boolean AND mask.

    A node is True only if it is True in all the masks.

    Attributes
    ----------
    masks : str | BooleanBaseNodeAttribute | list[str | BooleanBaseNodeAttribute]
        Masks to combine. Each mask is either the name of a boolean attribute already registered in the nodes,
        or a boolean node attribute builder.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor
        Computes the AND mask for the nodes `nodes_name` of the graph.
    """

    def _reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor:
        return torch.all(masks, dim=0)


class BooleanOrMask(BooleanOperation):
    """Boolean OR mask.

    A node is True if it is True in any of the masks.

    Attributes
    ----------
    masks : str | BooleanBaseNodeAttribute | list[str | BooleanBaseNodeAttribute]
        Masks to combine. Each mask is either the name of a boolean attribute already registered in the nodes,
        or a boolean node attribute builder.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph: HeteroData, nodes_name: str, **kwargs) -> torch.Tensor
        Computes the OR mask for the nodes `nodes_name` of the graph.
    """

    def _reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor:
        return torch.any(masks, dim=0)
