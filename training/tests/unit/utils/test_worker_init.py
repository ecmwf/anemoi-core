# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the blosc thread setting of dataloader workers."""

import multiprocessing
from collections.abc import Iterator
from functools import partial

import numcodecs.blosc
import numpy as np
import pytest
import torch
from numcodecs import Blosc
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from anemoi.training.utils.worker_init import set_blosc_threads
from anemoi.training.utils.worker_init import worker_init_func


@pytest.fixture
def restore_blosc() -> Iterator[None]:
    use_threads, nthreads = numcodecs.blosc.use_threads, numcodecs.blosc.get_nthreads()
    yield
    numcodecs.blosc.use_threads = use_threads
    numcodecs.blosc.set_nthreads(nthreads)


@pytest.mark.usefixtures("restore_blosc")
def test_none_does_not_change_blosc() -> None:
    numcodecs.blosc.use_threads = False
    set_blosc_threads(None)
    assert numcodecs.blosc.use_threads is False


@pytest.mark.usefixtures("restore_blosc")
def test_sets_blosc_threads() -> None:
    set_blosc_threads(3)
    assert numcodecs.blosc.use_threads is True
    assert numcodecs.blosc.get_nthreads() == 3


class BloscSettings(Dataset):
    """Each item compresses an array and returns the blosc settings of the worker."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        value = np.arange(100_000, dtype=np.float32) + index
        codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        assert np.array_equal(np.frombuffer(codec.decode(codec.encode(value)), dtype=np.float32), value)
        return torch.tensor([int(bool(numcodecs.blosc.use_threads)), numcodecs.blosc.get_nthreads()])

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        pass


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_workers_use_the_blosc_threads(context: str) -> None:
    loader = DataLoader(
        BloscSettings(),
        batch_size=None,
        num_workers=2,
        worker_init_fn=partial(worker_init_func, blosc_threads=4),
        multiprocessing_context=multiprocessing.get_context(context),
    )
    for settings in loader:
        assert settings.tolist() == [1, 4]
