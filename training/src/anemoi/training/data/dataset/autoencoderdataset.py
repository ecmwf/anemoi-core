# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from .singledataset import NativeGridDataset

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from anemoi.training.data.grid_indices import BaseGridIndices


class AutoencoderNativeGridDataset(NativeGridDataset):
    """Iterable autoencoder dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        """
        super().__init__(data_reader, grid_indices, relative_date_indices, timestep, shuffle, label)

    def __iter__(self) -> torch.Tensor:
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
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:

            i = int(i)
            grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)

            x = self.data[i : i + 1]  # trick to keep the date dimention
            x = x[..., grid_shard_indices]  # select the grid shard

            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            self.ensemble_dim = 1

            yield torch.from_numpy(x)
