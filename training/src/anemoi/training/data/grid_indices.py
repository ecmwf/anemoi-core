# (C) Copyright 2024 Anemoi contributors.
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
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)

ArrayIndex = slice | int | Sequence[int]


class BaseGridIndices(ABC):
    """Base class for custom grid indices."""

    def __init__(self, nodes_name: str, reader_group_size: int) -> None:
        self.nodes_name = nodes_name
        self.reader_group_size = reader_group_size

    def setup(self, graph: HeteroData) -> None:
        self.grid_size = self.compute_grid_size(graph)

    def get_shard_slice(self, reader_group_rank: int) -> slice:
        """Get the grid shard slice according to the reader rank."""
        assert (
            0 <= reader_group_rank < self.reader_group_size
        ), f"Invalid reader group rank {reader_group_rank}, expected in [0, {self.reader_group_size})"
        start = sum(self.shard_shapes[:reader_group_rank])
        end = start + self.shard_shapes[reader_group_rank]

        return slice(start, end)

    @cached_property
    def shard_shapes(self) -> list:
        # equivalent to torch.tensor_split(torch.arange(self.grid_size), self.reader_group_size)
        mod = self.grid_size % self.reader_group_size
        return [self.grid_size // self.reader_group_size + 1] * mod + [self.grid_size // self.reader_group_size] * (
            self.reader_group_size - mod
        )

    @property
    def supporting_arrays(self) -> dict:
        return {}

    @abstractmethod
    def compute_grid_size(self, graph: HeteroData) -> int: ...

    @abstractmethod
    def get_shard_indices(self, reader_group_rank: int) -> ArrayIndex: ...


class FullGrid(BaseGridIndices):
    """The full grid is loaded."""

    def compute_grid_size(self, graph: HeteroData) -> int:
        return graph[self.nodes_name].num_nodes

    def get_shard_indices(self, reader_group_rank: int) -> slice:
        return self.get_shard_slice(reader_group_rank)


class MaskedGrid(BaseGridIndices):
    """Grid is masked based on a node attribute."""

    def __init__(self, nodes_name: str, reader_group_size: int, node_attribute_name: str):
        super().__init__(nodes_name, reader_group_size)
        self.node_attribute_name = node_attribute_name

    def setup(self, graph: HeteroData) -> None:
        LOGGER.info(
            "The graph attribute %s of the %s nodes will be used to masking the spatial dimension.",
            self.node_attribute_name,
            self.nodes_name,
        )
        self.grid_indices = graph[self.nodes_name][self.node_attribute_name].squeeze().tolist()
        super().setup(graph)

    @property
    def supporting_arrays(self) -> dict:
        return {"grid_indices": np.array(self.grid_indices, dtype=np.int64)}

    def compute_grid_size(self, _graph: HeteroData) -> int:
        return len(self.grid_indices)

    def get_shard_indices(self, reader_group_rank: int) -> ArrayIndex:
        sequence_indices = self.get_shard_slice(reader_group_rank)
        return self.grid_indices[sequence_indices]
