# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
import os
import random
from functools import cached_property
from typing import Callable
from pathlib import Path
import numpy as np
import torch
import time
from einops import rearrange
from icecream import ic
from torch.utils.data import IterableDataset, get_worker_info

# from torch_geometric.nn import radius
import scipy.spatial

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.data.dataset.singledataset import NativeGridDataset
from anemoi.training.utils.seeding import get_base_seed

LOGGER = logging.getLogger(__name__)


class DownscalingDataset(NativeGridDataset):
    """Iterable dataset for Anemoi data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        lres_grid_indices: type[BaseGridIndices],
        hres_grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:

        super().__init__(
            data_reader=data_reader,
            grid_indices=hres_grid_indices,
            relative_date_indices=relative_date_indices,
            timestep=timestep,
            shuffle=shuffle,
            label=label,
        )
        self.lres_grid_indices = lres_grid_indices
        self.hres_grid_indices = hres_grid_indices
        LOGGER.info("dataset %s len valid date indices %d", label, len(self.valid_date_indices))

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:

            lres_grid_shard_indices = self.lres_grid_indices.get_shard_indices(self.reader_group_rank)
            hres_grid_shard_indices = self.hres_grid_indices.get_shard_indices(self.reader_group_rank)

            # Load full grid in CPU memory, select grid_shard after
            # Note that anemoi-datasets currently doesn't support slicing + indexing
            # in the same operation.
            x_in_lres, x_in_hres, y = self.data[i : i + 1]
            x_in_lres = x_in_lres[..., lres_grid_shard_indices]
            x_in_hres = x_in_hres[..., hres_grid_shard_indices]
            y = y[..., hres_grid_shard_indices]

            # iterate input
            x_in_hres = rearrange(
                x_in_hres,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )
            x_in_hres = torch.from_numpy(x_in_hres)

            x_in_lres = rearrange(
                x_in_lres,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )
            x_in_lres = torch.from_numpy(x_in_lres)

            y = rearrange(
                y,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )
            y = torch.from_numpy(y)

            # ic("data load time", time_end - time_start)

            yield x_in_lres, x_in_hres, y
