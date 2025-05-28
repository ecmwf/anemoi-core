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
        from anemoi.datasets import open_dataset

        assert nodes["node_type"] in [
            "ZarrDatasetNodes",
            "AnemoiDatasetNodes",
        ], f"{self.__class__.__name__} can only be used with AnemoiDatasetNodes."
        ds = open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()
        return torch.from_numpy(~np.isnan(ds))


class CutOutMask(BooleanBaseNodeAttribute):
    """Cut out mask."""

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        from anemoi.datasets import open_dataset

        assert "_dataset" in nodes and isinstance(
            nodes["_dataset"], dict
        ), "The '_dataset' attribute must be a dictionary."
        assert "cutout" in nodes["_dataset"], "The 'dataset' attribute must contain a 'cutout' key."
        num_lam, num_other = open_dataset(nodes["_dataset"]).grids
        return torch.tensor([True] * num_lam + [False] * num_other, dtype=torch.bool)
