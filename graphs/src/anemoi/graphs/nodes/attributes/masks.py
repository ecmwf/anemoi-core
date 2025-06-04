# (C) Copyright 2024- Anemoi contributors.
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
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.datasets import open_dataset
from anemoi.graphs.nodes.attributes.base_attributes import BooleanBaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class NonmissingAnemoiDatasetVariable(BooleanBaseNodeAttribute):
    """Mask of valid (not missing) values of a Anemoi dataset variable.

    It reads a variable from a Anemoi dataset and returns a boolean mask of nonmissing values in the first timestep.

    Attributes
    ----------
    variable : str
        Variable to read from the Anemoi dataset.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self, variable: str) -> None:
        super().__init__()
        self.variable = variable

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert nodes["node_type"] in [
            "ZarrDatasetNodes",
            "AnemoiDatasetNodes",
        ], f"{self.__class__.__name__} can only be used with AnemoiDatasetNodes."
        ds = open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()
        return torch.from_numpy(~np.isnan(ds))


class BaseCombineAnemoiDatasetsMask(BooleanBaseNodeAttribute):
    """Base class for computing mask based on anemoi-datasets combining operations."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert "_dataset" in nodes and isinstance(
            nodes["_dataset"], dict
        ), "The '_dataset' attribute must be a dictionary."
        grids_size = open_dataset(nodes["_dataset"]).grids
        assert (
            len(grids_size) == 2
        ), f"{self.__class__.__name__} is only supported for combining operations over 2 datasets."
        return torch.tensor([True] * grids_size[0] + [False] * grids_size[1], dtype=torch.bool)


class CutOutMask(BaseCombineAnemoiDatasetsMask):
    """Cut out mask."""


class GridsMask(BaseCombineAnemoiDatasetsMask):
    """Grids mask."""
