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
    """Base class for masks based on a variable of an Anemoi dataset.

    It reads the variable at the first date of the dataset. It can only be used with nodes built from an
    Anemoi dataset (i.e. :class:`AnemoiDatasetNodes`).

    The `_get_mask` method must be implemented by subclasses to define how the mask is computed from the
    variable values.

    Attributes
    ----------
    variable : str
        Variable to read from the Anemoi dataset.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.
    """

    def __init__(self, variable: str, name: str | None = None) -> None:
        super().__init__(name=name)
        self.variable = variable

    @abstractmethod
    def _get_mask(self, ds) -> np.ndarray: ...

    def _read_data(self, nodes: NodeStorage, **kwargs) -> np.ndarray:
        return open_dataset(nodes["_dataset"], select=self.variable)[0].squeeze()

    def _get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:

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
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph, nodes_name)
        Compute the mask for each node.
    """

    def __init__(self, variable: str, name: str | None = None) -> None:
        super().__init__(variable, name)
        self.variable = variable

    def _get_mask(self, ds) -> np.ndarray:
        return ~np.isnan(ds)


class NonzeroAnemoiDatasetVariable(BaseAnemoiDatasetVariable):
    """Mask of non-zero values of an Anemoi dataset variable.

    It reads a variable from an Anemoi dataset and returns a boolean mask of non-zero values in the first timestep.

    Attributes
    ----------
    variable : str
        Variable to read from the Anemoi dataset.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph, nodes_name)
        Compute the mask for each node.
    """

    def __init__(self, variable: str, name: str | None = None) -> None:
        super().__init__(variable, name)
        self.variable = variable

    def _get_mask(self, ds) -> np.ndarray:
        return ds != 0


class BaseCombineAnemoiDatasetsMask(BooleanBaseNodeAttribute, ABC):
    """Base class for masks based on anemoi-datasets combining operations (e.g. cutout, grids).

    The nodes must be built from an Anemoi dataset combining several grids. The mask is True for the nodes
    belonging to the grids listed in `grids`.

    Attributes
    ----------
    grids : list[int]
        Positions of the grids whose nodes are set to True. It must be set by subclasses.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.
    """

    grids: list[int] | None = None

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        if self.grids is None:
            raise AttributeError(f"{self.__class__.__name__} class must set 'grids' attribute.")

    def _get_grid_sizes(self, nodes):
        from anemoi.datasets import open_dataset

        assert "_dataset" in nodes and isinstance(
            nodes["_dataset"], (dict, str)
        ), "The '_dataset' attribute must be a dictionary or string."

        return open_dataset(nodes["_dataset"]).grids

    @staticmethod
    def _get_mask_from_grid_sizes(grid_sizes: tuple[int], masked_grids_posisitons: list[int]):
        assert isinstance(masked_grids_posisitons, list), "masked_grids_positions must be a list"
        assert min(masked_grids_posisitons) >= 0, "masked_grids_positions must be non-negative"
        assert max(masked_grids_posisitons) < len(grid_sizes), f"masked_grids_positions must be < {len(grid_sizes)}"
        mask = torch.zeros(sum(grid_sizes), dtype=torch.bool)
        for grid_id in masked_grids_posisitons:
            mask[sum(grid_sizes[:grid_id]) : sum(grid_sizes[: grid_id + 1])] = True
        return mask

    def _get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        grid_sizes = self._get_grid_sizes(nodes)
        return BaseCombineAnemoiDatasetsMask._get_mask_from_grid_sizes(grid_sizes, self.grids)


class CutOutMask(BaseCombineAnemoiDatasetsMask):
    """Cut out mask.

    It computes a mask for the first dataset in the cutout operation, i.e. it sets to True the nodes of the
    first (index 0) grid.

    Attributes
    ----------
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph, nodes_name)
        Compute the mask for each node.
    """

    def __init__(self, name: str | None = None) -> None:
        self.grids = [0]  # It sets as true the nodes from the first (index=0) grid
        super().__init__(name=name)


class GridsMask(BaseCombineAnemoiDatasetsMask):
    """Grids mask.

    It computes a mask that sets to True the nodes of the selected grids of an Anemoi dataset combining
    several grids (e.g. with the cutout or grids operations).

    Attributes
    ----------
    grids : int | list[int], optional
        Grid positions to set as True. Defaults to 0, which sets True only the nodes from the first dataset.
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph, nodes_name)
        Compute the mask for each node.
    """

    def __init__(self, grids: int | list[int] = 0, name: str | None = None) -> None:
        self.grids = [grids] if isinstance(grids, int) else grids
        super().__init__(name=name)


class LimitedAreaMask(BooleanBaseNodeAttribute):
    """Limited area mask.

    It adds a mask based on an area of interest. This mask is only defined
    for nodes built with a subclass of `StretchedIcosahedronNodes`.

    Attributes
    ----------
    name : str
        The name of the node attribute that will be used to store the computed values in the :class:`HeteroData` graph.

    Methods
    -------
    compute(graph, nodes_name)
        Compute the mask for each node.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)

    def _get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        assert nodes["node_type"] in [
            "StretchedTriNodes"
        ], f"{self.__class__.__name__} can only be used with StretchedIcosahedronNodes."
        lam_mask = nodes["_area_mask_builder"].get_mask(nodes.x)
        return lam_mask
