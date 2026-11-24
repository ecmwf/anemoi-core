# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for cache records that are the compressed chunks of the source zarr."""

from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc
from numcodecs import Zstd

from anemoi.training.utils.dataset_cache import DatasetCache
from anemoi.training.utils.dataset_cache import DatasetCacheNamespace
from anemoi.training.utils.dataset_cache import SourceChunks
from anemoi.training.utils.dataset_cache import decode_cache_array

SHAPE = (6, 3, 1, 40)  # dates, variables, ensemble, grid points


# The classes have the names of the anemoi-datasets classes that SourceChunks accepts.
class GriddedZarr:
    def __init__(self, array: zarr.Array, missing: set[int] | None = None) -> None:
        self.data = array
        self.missing = missing or set()


class Subset:
    def __init__(self, dataset: object, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices


class Reader:
    def __init__(self, data: object, length: int) -> None:
        self.data = data
        self.num_sequences = 1
        self.length = length

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        return self.length


def make_array(tmp_path: Path, **kwargs: object) -> tuple[zarr.Array, np.ndarray]:
    values = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float32)
    options = {"chunks": (1, *SHAPE[1:]), "compressor": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)}
    options.update(kwargs)
    array = zarr.open(str(tmp_path / "data.zarr"), mode="w", shape=SHAPE, dtype="f4", **options)
    array[:] = values
    return array, values


def make_cache(tmp_path: Path, data: object, length: int) -> tuple[DatasetCache, str]:
    cache = DatasetCache(object(), tmp_path / "cache")
    cache.node_id, cache.budget = 0, None
    namespace = DatasetCacheNamespace(tmp_path / "cache", "data:test", Reader(data, length))
    cache.namespaces[namespace.dataset_id] = namespace
    return cache, namespace.dataset_id


def test_records_are_the_source_chunks(tmp_path: Path) -> None:
    array, values = make_array(tmp_path)
    cache, dataset_id = make_cache(tmp_path, GriddedZarr(array), SHAPE[0])

    records = cache.read_source_records(dataset_id, 0, [1, 4])

    np.testing.assert_array_equal(np.stack(records), values[[1, 4]])
    namespace = cache.namespaces[dataset_id]
    for position in (1, 4):
        entry, _ = namespace.paths(0, position)
        payload = entry.read_bytes()
        # the record is the .npy header and the chunk bytes, with no other compression
        assert payload.endswith(array.chunk_store[array._chunk_key((position, 0, 0, 0))])
        np.testing.assert_array_equal(decode_cache_array(payload), values[position])
        np.testing.assert_array_equal(namespace.local(0, position), values[position])
    assert cache.cache_source_records.value == 2


def test_date_subset_maps_positions_to_zarr_indices(tmp_path: Path) -> None:
    array, values = make_array(tmp_path)
    # a subset of a subset: positions 0, 1, 2 are zarr indices 5, 3, 1
    data = Subset(Subset(GriddedZarr(array), [1, 2, 3, 4, 5]), [4, 2, 0])
    cache, dataset_id = make_cache(tmp_path, data, 3)

    records = cache.read_source_records(dataset_id, 0, [0, 1, 2])

    np.testing.assert_array_equal(np.stack(records), values[[5, 3, 1]])


@pytest.mark.parametrize("grid_indices", [None, slice(None), slice(0, SHAPE[-1]), slice(0, None, 1)])
def test_full_grid_uses_source_chunks(tmp_path: Path, grid_indices: slice | None) -> None:
    array, values = make_array(tmp_path)
    cache, dataset_id = make_cache(tmp_path, GriddedZarr(array), SHAPE[0])
    records = cache.read_source_records(dataset_id, 0, [2], grid_indices)
    np.testing.assert_array_equal(records[0], values[2])


@pytest.mark.parametrize("grid_indices", [slice(0, 20), slice(None, None, 2), np.arange(SHAPE[-1])])
def test_grid_shard_does_not_use_source_chunks(tmp_path: Path, grid_indices: object) -> None:
    array, _ = make_array(tmp_path)
    cache, dataset_id = make_cache(tmp_path, GriddedZarr(array), SHAPE[0])
    assert cache.read_source_records(dataset_id, 0, [2], grid_indices) is None


@pytest.mark.parametrize(
    "options",
    [
        {"compressor": Zstd()},
        {"compressor": None},
        {"chunks": (2, *SHAPE[1:])},
        {"chunks": (1, 1, 1, SHAPE[-1])},
        {"order": "F"},
    ],
)
def test_other_zarr_layouts_do_not_use_source_chunks(tmp_path: Path, options: dict) -> None:
    array, _ = make_array(tmp_path, **options)
    assert SourceChunks.from_dataset(GriddedZarr(array)) is None


def test_other_datasets_do_not_use_source_chunks(tmp_path: Path) -> None:
    array, _ = make_array(tmp_path)

    class Select:  # for example a selection of variables
        def __init__(self, dataset: object) -> None:
            self.dataset, self.indices = dataset, [0, 1]

    assert SourceChunks.from_dataset(Select(GriddedZarr(array))) is None
    assert SourceChunks.from_dataset(np.zeros(SHAPE)) is None
    assert SourceChunks.from_dataset(None) is None


def test_missing_date_or_chunk_falls_back(tmp_path: Path) -> None:
    array, _ = make_array(tmp_path)
    cache, dataset_id = make_cache(tmp_path, GriddedZarr(array, missing={3}), SHAPE[0])
    assert cache.read_source_records(dataset_id, 0, [2, 3]) is None

    del array.chunk_store[array._chunk_key((4, 0, 0, 0))]
    assert cache.read_source_records(dataset_id, 0, [4]) is None
    # nothing was written for the request that fell back
    assert not cache.namespaces[dataset_id].paths(0, 2)[0].exists()


def test_partition_stores_only_owned_source_records(tmp_path: Path) -> None:
    array, values = make_array(tmp_path)
    cache, dataset_id = make_cache(tmp_path, GriddedZarr(array), SHAPE[0])
    cache.partition, cache.num_nodes = True, 2  # node 0 owns dates 0-2 of 6

    records = cache.read_source_records(dataset_id, 0, [1, 4])

    # the sample has all the records, but only the owned record is in the cache
    np.testing.assert_array_equal(np.stack(records), values[[1, 4]])
    namespace = cache.namespaces[dataset_id]
    assert namespace.paths(0, 1)[0].exists()
    assert not namespace.paths(0, 4)[0].exists()
    assert cache.cache_not_owned.value == 1
