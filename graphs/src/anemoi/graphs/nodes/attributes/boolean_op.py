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
    """Base class for boolean operations."""

    def __init__(self, masks: MaskAttributeType | list[MaskAttributeType]) -> None:
        super().__init__()
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

        return mask.get_raw_values(nodes, **kwargs)

    @abstractmethod
    def reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor: ...

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        mask_values = [BooleanOperation.get_mask_values(mask, nodes, **kwargs) for mask in self.masks]
        return self.reduce_op(torch.stack(mask_values))


class BooleanNot(BooleanOperation):
    """Boolean NOT mask."""

    def reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor:
        assert len(self.masks) == 1, f"The {self.__class__.__name__} can only be aplied to one mask."
        return ~masks[0]


class BooleanAndMask(BooleanOperation):
    """Boolean AND mask."""

    def reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor:
        return torch.all(masks, dim=0)


class BooleanOrMask(BooleanOperation):
    """Boolean OR mask."""

    def reduce_op(self, masks: list[torch.Tensor]) -> torch.Tensor:
        return torch.any(masks, dim=0)
