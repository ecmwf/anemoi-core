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

import numpy as np
import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.datasets import open_dataset
from anemoi.graphs.nodes.attributes.base_attributes import BooleanBaseNodeAttribute

LOGGER = logging.getLogger(__name__)


class BaseAnemoiDatasetVariable(BooleanBaseNodeAttribute):
    """Base class for computing mask based on a variable in an Anemoi dataset."""

    def __init__(self, variable: str) -> None:
        super().__init__()
        self.variable = variable

    @abstractmethod
    def _get_mask(self, ds) -> np.ndarray: ...

    def _read_data(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        return open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:

        assert nodes["node_type"] in [
            "ZarrDatasetNodes",
            "AnemoiDatasetNodes",
        ], f"{self.__class__.__name__} can only be used with AnemoiDatasetNodes."
        ds = self._read_data(nodes)
        return torch.from_numpy(self._get_mask(ds))


class NonmissingAnemoiDatasetVariable(BaseAnemoiDatasetVariable):
    """Mask of valid (not missing) values of an Anemoi dataset variable.

    It reads a variable from an Anemoi dataset and returns a boolean mask of nonmissing values in the first timestep.

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
        super().__init__(variable)
        self.variable = variable

    def _get_mask(self, ds) -> np.ndarray:
        return ~np.isnan(ds)


class NonzeroAnemoiDatasetVariable(BaseAnemoiDatasetVariable):
    """Mask of non-zero values of an Anemoi dataset variable.

    Reads a variable from an Anemoi dataset and returns a boolean mask of non-zero values in the first timestep.

    Attributes
    ----------
    variable : str
        Variable to read from the Anemoi dataset.

    Methods
    -------
    compute(self, graph, nodes_name)
        Computer the attribute for each node.
    """

    def __init__(self, variable: str) -> None:
        super().__init__(variable)
        self.variable = variable

    def _get_mask(self, ds) -> np.ndarray:
        return ds != 0


class BaseCombineAnemoiDatasetsMask(BooleanBaseNodeAttribute, ABC):
    """Base class for computing mask based on anemoi-datasets combining operations."""

    grids: list[int] | None = None

    def __init__(self) -> None:
        super().__init__()
        if self.grids is None:
            raise AttributeError(f"{self.__class__.__name__} class must set 'grids' attribute.")

    def get_grid_sizes(self, nodes):
        from anemoi.datasets import open_dataset

        assert "_dataset" in nodes and isinstance(
            nodes["_dataset"], (dict, str)
        ), "The '_dataset' attribute must be a dictionary or string."

        return open_dataset(nodes["_dataset"]).grids

    @staticmethod
    def get_mask_from_grid_sizes(grid_sizes: tuple[int], masked_grids_posisitons: list[int]):
        assert isinstance(masked_grids_posisitons, list), "masked_grids_positions must be a list"
        assert min(masked_grids_posisitons) >= 0, "masked_grids_positions must be non-negative"
        assert max(masked_grids_posisitons) < len(grid_sizes), f"masked_grids_positions must be < {len(grid_sizes)}"
        mask = torch.zeros(sum(grid_sizes), dtype=torch.bool)
        for grid_id in masked_grids_posisitons:
            mask[sum(grid_sizes[:grid_id]) : sum(grid_sizes[: grid_id + 1])] = True
        return mask

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        grid_sizes = self.get_grid_sizes(nodes)
        return BaseCombineAnemoiDatasetsMask.get_mask_from_grid_sizes(grid_sizes, self.grids)


class CutOutMask(BaseCombineAnemoiDatasetsMask):
    """Cut out mask.

    It computes a mask for the first dataset in the cutout operation.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def __init__(self) -> None:
        self.grids = [0]  # It sets as true the nodes from the first (index=0) grid
        super().__init__()


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
        self.grids = [grids] if isinstance(grids, int) else grids
        super().__init__()


class LimitedAreaMask(BooleanBaseNodeAttribute):
    """Limited area mask.

    It adds a mask based on an area of interest. This mask is only defined
    for nodes built with a subclass of `StretchedIcosahedronNodes`.

    Methods
    -------
    compute(self, graph, nodes_name)
        Compute the attribute for each node.
    """

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert nodes["node_type"] in [
            "StretchedTriNodes"
        ], f"{self.__class__.__name__} can only be used with StretchedIcosahedronNodes."
        lam_mask = nodes["_area_mask_builder"].get_mask(nodes.x)
        return torch.from_numpy(lam_mask)
