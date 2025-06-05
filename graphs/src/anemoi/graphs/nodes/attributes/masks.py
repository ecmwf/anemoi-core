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

    def get_grid_sizes(self, nodes):
        assert "_dataset" in nodes and isinstance(
            nodes["_dataset"], dict
        ), "The '_dataset' attribute must be a dictionary."
        return open_dataset(nodes["_dataset"]).grids


class CutOutMask(BaseCombineAnemoiDatasetsMask):
    """Cut out mask.

    It computes a mask for the first dataset in the cutout operation.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        grid_sizes = self.get_grid_sizes(nodes)
        return torch.tensor([True] * grid_sizes[0] + [False] * grid_sizes[1], dtype=torch.bool)


class GridsMask(BaseCombineAnemoiDatasetsMask):
    """Grids mask.

    It reads a variable from a Anemoi dataset and returns a boolean mask of nonmissing values in the first timestep.

    Attributes
    ----------
    grids : int | list[int], optional
        Grid positions to set as True. Defaults to 0, which sets True only the nodes from the first dataset.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self, grids: int | list[int] = 0) -> None:
        super().__init__()
        self.grids = [grids] if isinstance(grids, int) else grids

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        grid_sizes = self.get_grid_sizes(nodes)
        mask = torch.zeros(sum(grid_sizes), dtype=torch.bool)
        for grid_id in self.grids:
            start_grid = 0 if grid_id == 0 else grid_sizes[grid_id - 1]
            end_grid = grid_sizes[grid_id - 1]
            mask[start_grid:end_grid] = True
        return mask
