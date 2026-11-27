# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for anemoi.training.utils.dataset_cache."""

import errno
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.training.utils import cache_format
from anemoi.training.utils.cache_format import FULL_GRID
from anemoi.training.utils.cache_transport import CacheServer
from anemoi.training.utils.cache_transport import RemoteCacheError
from anemoi.training.utils.dataset_cache import CacheFullError
from anemoi.training.utils.dataset_cache import DatasetCache
from anemoi.training.utils.dataset_cache import DatasetCacheNamespace
from anemoi.training.utils.dataset_cache import _dataset_id

DATASET_ID = "analysis:fingerprint"


class FakeReader:
    """Analysis data (dates, variables, ensemble, gridpoints), or trajectory data with one more dimension first."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    @property
    def num_sequences(self) -> int:
        return 1 if self.data.ndim == 4 else self.data.shape[0]

    def sequence_length(self, sequence: int = 0) -> int:
        del sequence
        return self.data.shape[0] if self.data.ndim == 4 else self.data.shape[-2]

    @property
    def grid_size(self) -> int:
        return self.data.shape[-1]


@pytest.fixture
def sample_data() -> np.ndarray:
    """(dates=10, variables=3, ensemble=2, gridpoints=5)."""
    return np.random.default_rng(42).standard_normal((10, 3, 2, 5)).astype(np.float32)


def make_cache(cache_path: Path, data: np.ndarray, node_id: int = 0, num_nodes: int = 1) -> DatasetCache:
    """Return a cache of one node, without torch distributed."""
    cache = DatasetCache(cache_path)
    cache.cache_path, cache.min_free_bytes, cache.global_rank = cache_path, 0, 0
    cache.node_id, cache.num_nodes = node_id, num_nodes
    cache.namespaces[DATASET_ID] = DatasetCacheNamespace(cache_path, DATASET_ID, FakeReader(data))
    return cache


def stored_positions(cache: DatasetCache, grid_id: str = FULL_GRID) -> list[int]:
    namespace = cache.namespaces[DATASET_ID]
    positions = range(namespace.sequence_lengths[0])
    return [position for position in positions if namespace.path(0, position, grid_id).exists()]


# ---------------------------------------------------------------------------
# Records of one dataset on one node
# ---------------------------------------------------------------------------


class TestDatasetCacheNamespace:
    def namespace(self, tmp_path: Path, data: np.ndarray) -> DatasetCacheNamespace:
        return DatasetCacheNamespace(tmp_path, DATASET_ID, FakeReader(data))

    def test_record_is_stored_once(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        namespace = self.namespace(tmp_path, sample_data)
        assert namespace.store(0, 3, sample_data[3], FULL_GRID, 0)
        assert not namespace.store(0, 3, np.zeros_like(sample_data[3]), FULL_GRID, 0)
        np.testing.assert_array_equal(namespace.local(0, 3, FULL_GRID), sample_data[3])
        assert namespace.local(0, 4, FULL_GRID) is None

    def test_trajectory_sequences_do_not_collide(self, tmp_path: Path) -> None:
        data = np.arange(2 * 3 * 1 * 4 * 5).reshape(2, 3, 1, 4, 5)
        namespace = self.namespace(tmp_path, data)
        namespace.store(0, 2, data[0, :, :, 2, :], FULL_GRID, 0)
        namespace.store(1, 2, data[1, :, :, 2, :], FULL_GRID, 0)
        np.testing.assert_array_equal(namespace.local(0, 2, FULL_GRID), data[0, :, :, 2, :])
        np.testing.assert_array_equal(namespace.local(1, 2, FULL_GRID), data[1, :, :, 2, :])

    def test_writer_that_loses_the_race_does_not_replace_the_record(
        self,
        tmp_path: Path,
        sample_data: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        namespace = self.namespace(tmp_path, sample_data)
        entry = namespace.path(0, 3, FULL_GRID)
        real_link = os.link

        def other_writer_first(source: Path, target: Path) -> None:
            target.write_bytes(cache_format.encode(sample_data[3]))  # another process wins the race
            real_link(source, target)

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.os.link", other_writer_first)
        assert not namespace.store(0, 3, np.zeros_like(sample_data[3]), FULL_GRID, 0)
        np.testing.assert_array_equal(namespace.local(0, 3, FULL_GRID), sample_data[3])
        assert not list(entry.parent.glob("*.tmp"))

    def test_disk_full_removes_partial_record(
        self,
        tmp_path: Path,
        sample_data: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        namespace = self.namespace(tmp_path, sample_data)

        def disk_full(_source: Path, _target: Path) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.os.link", disk_full)
        with pytest.raises(CacheFullError, match="No space left"):
            namespace.store(0, 3, sample_data[3], FULL_GRID, 0)
        entry = namespace.path(0, 3, FULL_GRID)
        assert not entry.exists()
        assert not list(entry.parent.glob("*.tmp"))

    def test_unrelated_write_error_is_not_cache_full(
        self,
        tmp_path: Path,
        sample_data: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        namespace = self.namespace(tmp_path, sample_data)

        def no_permission(_source: Path, _target: Path) -> None:
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.os.link", no_permission)
        with pytest.raises(PermissionError):
            namespace.store(0, 3, sample_data[3], FULL_GRID, 0)

    def test_store_keeps_free_space_on_the_disk(
        self,
        tmp_path: Path,
        sample_data: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        namespace = self.namespace(tmp_path, sample_data)
        record_bytes = len(cache_format.encode(sample_data[0]))
        disk = SimpleNamespace(total=10**6, used=0, free=1000 + record_bytes)
        monkeypatch.setattr("anemoi.training.utils.dataset_cache.shutil.disk_usage", lambda _path: disk)

        assert namespace.store(0, 0, sample_data[0], FULL_GRID, min_free_bytes=1000)  # leaves 1000 bytes free
        disk.free = 999 + record_bytes  # other files on the disk use the space
        with pytest.raises(CacheFullError, match="free on the disk"):
            namespace.store(0, 1, sample_data[1], FULL_GRID, min_free_bytes=1000)

    def test_records_are_compressed(self, tmp_path: Path) -> None:
        data = np.zeros((10, 3, 2, 500), dtype=np.float32)
        namespace = self.namespace(tmp_path, data)
        assert namespace.store(0, 3, data[3], FULL_GRID, 0)
        np.testing.assert_array_equal(namespace.local(0, 3, FULL_GRID), data[3])
        assert namespace.path(0, 3, FULL_GRID).stat().st_size < data[3].nbytes / 2

    @pytest.mark.parametrize("grid_indices", [None, slice(None), slice(0, 5), slice(0, None, 1)])
    def test_full_grid(self, tmp_path: Path, sample_data: np.ndarray, grid_indices: slice | None) -> None:
        # training gives slice(0, grid_size) for the full grid
        assert self.namespace(tmp_path, sample_data).grid_id(grid_indices) == FULL_GRID

    def test_grid_shards_have_different_ids(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        namespace = self.namespace(tmp_path, sample_data)
        ids = {namespace.grid_id(indices) for indices in (slice(0, 2), slice(2, 5), np.array([0, 3]))}
        assert len(ids) == 3
        assert FULL_GRID not in ids


# ---------------------------------------------------------------------------
# Set-up
# ---------------------------------------------------------------------------


class FakeData:
    """The parts of an anemoi-datasets dataset that _dataset_id uses."""

    def __init__(self, shape: tuple = (4, 2, 1, 5), tree: str = "GriddedZarr\n  path: /data/a.zarr") -> None:
        self.shape, self.ndim, self.dtype, self._tree = shape, len(shape), np.dtype("float32"), tree

    def tree(self) -> str:
        return self._tree


class IdentityReader:
    variables = ("a", "b")

    def __init__(self, metadata: dict) -> None:
        self.data = FakeData()
        self.metadata = metadata


def make_zarr(path: Path, first_date: str = "2023-01-01T00:00:00", dates: int = 16) -> str:
    """Write a small anemoi-datasets zarr: 6-hourly dates, 2 variables, 4 grid points."""
    import zarr

    variables, grid = ["a", "b"], 4
    root = zarr.open_group(str(path), mode="w", zarr_format=2) if zarr.__version__ >= "3" else zarr.open(str(path), "w")
    data = np.arange(dates * len(variables) * grid, dtype="f4").reshape(dates, len(variables), 1, grid)
    root.create_dataset("data", data=data, chunks=(1, len(variables), 1, grid))
    root.create_dataset("dates", data=np.datetime64(first_date) + np.arange(dates) * np.timedelta64(6, "h"))
    root.create_dataset("latitudes", data=np.linspace(-80, 80, grid))
    root.create_dataset("longitudes", data=np.linspace(0, 350, grid))
    for name in ("mean", "stdev", "maximum", "minimum", "sums", "squares", "count", "has_nans"):
        root.create_dataset(name, data=np.ones(len(variables)))
    end_date = np.datetime64(first_date) + (dates - 1) * np.timedelta64(6, "h")
    root.attrs.update(
        {
            "frequency": "6h",
            "resolution": "o96",
            "start_date": first_date,
            "end_date": str(end_date),
            "variables": variables,
            "missing_dates": [],
            "ensemble_dimension": 1,
            "field_shape": [grid],
            "flatten_grid": True,
            "version": "0.30",
        },
    )
    return str(path)


def test_dataset_id_separates_date_ranges_and_datasets(tmp_path: Path) -> None:
    from anemoi.training.data.readers import GriddedDataReader

    first, second = make_zarr(tmp_path / "first.zarr"), make_zarr(tmp_path / "second.zarr")
    day_1 = GriddedDataReader(dataset_config=first, start="2023-01-01", end="2023-01-01")
    day_3 = GriddedDataReader(dataset_config=first, start="2023-01-03", end="2023-01-03")
    other = GriddedDataReader(dataset_config=second, start="2023-01-01", end="2023-01-01")
    again = GriddedDataReader(dataset_config=first, start="2023-01-01", end="2023-01-01")
    assert day_1.data.shape == day_3.data.shape == other.data.shape  # the shape alone does not separate them

    ids = [_dataset_id("data", reader) for reader in (day_1, day_3, other, again)]
    assert len(set(ids[:3])) == 3  # other dates, or another dataset: other records
    assert ids[3] == ids[0]  # the same data opened in the same way: the same records


def test_dataset_id_does_not_use_metadata() -> None:
    # the metadata of a cloned dataset can change between processes (for example an invalid end_date)
    first = _dataset_id("data", IdentityReader({"end_date": "1970-01-01T00:00:00"}))
    second = _dataset_id("data", IdentityReader({"end_date": "170884930401-06-20T02:29:53"}))
    assert first == second


def test_ranks_with_different_dataset_ids_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = DatasetCache(tmp_path)
    monkeypatch.setattr(cache, "_gather", lambda value: [value, ["data:bbbb"]])
    with pytest.raises(RuntimeError, match="do not agree"):
        cache.attach({"data": IdentityReader({})})


class TestHostnameCheck:
    def cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known_hosts: set[str]) -> DatasetCache:
        cache = DatasetCache(tmp_path, hostname_suffix="-ib")
        cache.node_id, cache.hostnames = 0, ["node-a-ib", "node-b-ib"]

        def getaddrinfo(host: str, *_args: object) -> list:
            if host not in known_hosts:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.socket.getaddrinfo", getaddrinfo)
        monkeypatch.setattr(cache, "_gather", lambda value: [value, value])
        return cache

    def test_known_hostnames_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cache(tmp_path, monkeypatch, {"node-a-ib", "node-b-ib"})._check_hostnames()

    def test_own_hostname_is_not_checked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cache(tmp_path, monkeypatch, {"node-b-ib"})._check_hostnames()

    def test_unknown_hostname_stops_the_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(RuntimeError, match=r"node-b-ib.*hostname_suffix \('-ib'\)"):
            self.cache(tmp_path, monkeypatch, set())._check_hostnames()

    def test_all_ranks_stop_when_one_rank_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = self.cache(tmp_path, monkeypatch, {"node-b-ib"})
        monkeypatch.setattr(cache, "_gather", lambda value: [value, ["node-a-ib (Name or service not known)"]])
        with pytest.raises(RuntimeError, match="node-a-ib"):
            cache._check_hostnames()


def test_setup_keeps_a_fraction_of_the_disk_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    disk = SimpleNamespace(total=1000 * 10**9, used=0, free=1000 * 10**9)
    monkeypatch.setattr("anemoi.training.utils.dataset_cache.shutil.disk_usage", lambda _path: disk)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        cache = DatasetCache(tmp_path / "missing" / "root", max_fraction=0.9)
        cache.setup()  # makes the cache root
        try:
            assert cache.min_free_bytes == 100 * 10**9
            assert cache.cache_path.is_dir()
            assert cache.is_node_leader
            assert cache.ports == [cache.server.port]
        finally:
            cache.teardown()
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Reads and writes
# ---------------------------------------------------------------------------


def test_check_cache_returns_hits_and_missing_indices(tmp_path: Path, sample_data: np.ndarray) -> None:
    cache = make_cache(tmp_path, sample_data)
    cache.store_records(DATASET_ID, 0, [3], sample_data[[3]])

    values, missing = cache.check_cache(DATASET_ID, 0, [3, 4])

    np.testing.assert_array_equal(values[0], sample_data[3])
    assert values[1] is None
    assert missing == [1]
    assert (cache.cache_hits_local.value, cache.cache_misses.value) == (1, 1)


def test_each_date_has_one_owner_node(tmp_path: Path, sample_data: np.ndarray) -> None:
    caches = [make_cache(tmp_path / str(node), sample_data, node, num_nodes=3) for node in range(3)]
    owners = [caches[0].owner(caches[0].namespaces[DATASET_ID], 0, position) for position in range(10)]
    assert owners == sorted(owners)  # contiguous parts, in node order
    assert set(owners) == {0, 1, 2}
    assert all(cache.owner(cache.namespaces[DATASET_ID], 0, 7) == owners[7] for cache in caches)


def test_node_stores_only_its_part_of_full_grid_records(tmp_path: Path, sample_data: np.ndarray) -> None:
    cache = make_cache(tmp_path, sample_data, node_id=0, num_nodes=2)
    cache.store_records(DATASET_ID, 0, list(range(10)), sample_data)
    assert stored_positions(cache) == [0, 1, 2, 3, 4]


def test_node_stores_all_grid_shard_records_that_it_reads(tmp_path: Path, sample_data: np.ndarray) -> None:
    # with model sharding, the ranks that read a shard can all be on one node: no other node caches it
    cache = make_cache(tmp_path, sample_data, node_id=0, num_nodes=2)
    shard = slice(0, 2)
    cache.store_records(DATASET_ID, 0, [7, 8], sample_data[[7, 8], ..., shard], shard)
    assert stored_positions(cache, cache.namespaces[DATASET_ID].grid_id(shard)) == [7, 8]


def test_grid_shard_records_are_not_read_from_other_nodes(
    tmp_path: Path,
    sample_data: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_cache(tmp_path, sample_data, node_id=0, num_nodes=2)
    monkeypatch.setattr(cache, "_request", lambda *_args: pytest.fail("a shard record was requested"))
    values, missing = cache.check_cache(DATASET_ID, 0, [7], slice(0, 2))
    assert (values, missing) == ([None], [0])


def test_missing_records_are_requested_from_their_owner(
    tmp_path: Path,
    sample_data: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_cache(tmp_path, sample_data, node_id=0, num_nodes=2)
    requests = []

    def request(node: int, _dataset_id: str, _sequence: int, _grid_id: str, positions: list[int]) -> list:
        requests.append((node, positions))
        return [sample_data[7], None]  # the owner does not have date 8 yet

    monkeypatch.setattr(cache, "_request", request)
    values, missing = cache.check_cache(DATASET_ID, 0, [2, 7, 8])  # node 0 owns date 2

    assert requests == [(1, [7, 8])]
    np.testing.assert_array_equal(values[1], sample_data[7])
    assert missing == [0, 2]
    assert (cache.cache_hits_remote.value, cache.cache_misses.value) == (1, 2)


def test_unavailable_node_gives_misses(
    tmp_path: Path,
    sample_data: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = make_cache(tmp_path, sample_data, node_id=0, num_nodes=2)
    cache.hostnames, cache.ports = ["node-a", "node-b"], [0, 0]

    def unavailable(*_args: object) -> list:
        msg = "timeout"
        raise RemoteCacheError(msg)

    monkeypatch.setattr("anemoi.training.utils.dataset_cache.CacheClient.request", unavailable)
    values, missing = cache.check_cache(DATASET_ID, 0, [7, 8])

    assert (values, missing) == ([None, None], [0, 1])
    assert cache.cache_remote_errors.value == 1
    assert "Remote cache read from node 1 failed: timeout" in caplog.text


def test_full_cache_stays_full(
    tmp_path: Path,
    sample_data: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = make_cache(tmp_path, sample_data)
    record_bytes = len(cache_format.encode(sample_data[0]))
    disk = SimpleNamespace(total=10**6, used=0, free=2 * record_bytes)
    monkeypatch.setattr("anemoi.training.utils.dataset_cache.shutil.disk_usage", lambda _path: disk)

    def write(positions: list[int]) -> None:
        for position in positions:
            cache.store_records(DATASET_ID, 0, [position], sample_data[[position]])
            disk.free -= record_bytes

    write([0, 1, 2])  # no exception
    assert cache.cache_full.value
    assert "Dataset cache on node 0 is full" in caplog.text

    disk.free = 10**6  # space is free again, but the cache does not take more records
    write([3])
    assert stored_positions(cache) == [0, 1]


def test_remote_read_from_a_forked_dataloader_worker(tmp_path: Path, sample_data: np.ndarray) -> None:
    """Node 0 reads the records of node 1 from its server, in a persistent forked worker."""
    from torch.utils.data import DataLoader
    from torch.utils.data import IterableDataset

    node_1 = make_cache(tmp_path / "node-1", sample_data, node_id=1, num_nodes=2)
    node_1.store_records(DATASET_ID, 0, [7, 8], sample_data[[7, 8]])
    server = CacheServer(node_1.cache_path)
    server.start()
    node_0 = make_cache(tmp_path / "node-0", sample_data, node_id=0, num_nodes=2)
    node_0.hostnames, node_0.ports = ["127.0.0.1", "127.0.0.1"], [0, server.port]

    class Reads(IterableDataset):
        def __iter__(self):
            values, missing = node_0.check_cache(DATASET_ID, 0, [7, 8, 9])
            yield np.stack(values[:2]), missing

    loader = DataLoader(
        Reads(),
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    try:
        for _ in range(2):  # two epochs in the same worker process
            ((values, missing),) = list(loader)
            np.testing.assert_array_equal(values, sample_data[[7, 8]])
            assert missing == [2]  # node 1 does not have date 9
        assert node_0.cache_hits_remote.value == 4
    finally:
        del loader
        server.close()


# ---------------------------------------------------------------------------
# Datamodule
# ---------------------------------------------------------------------------


class AttachableReader(FakeReader):
    variables = ("a", "b", "c")

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        self.data = FakeData(data.shape, tree=f"Subset\n  dates: {data.shape[0]}\n  GriddedZarr")
        self.cache = None

    def set_cache(self, cache: DatasetCache, dataset_id: str) -> None:
        self.cache = (cache, dataset_id)


@pytest.mark.parametrize("cache_dir", [None, "cache"])
def test_datamodule_starts_and_stops_the_cache(tmp_path: Path, sample_data: np.ndarray, cache_dir: str | None) -> None:
    import torch.distributed as dist
    from omegaconf import OmegaConf

    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    datamodule = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    datamodule.cache = None
    hardware = {"cache_max_fraction": 0.9, "hostname_suffix": None}
    if cache_dir is not None:
        hardware["cache_dir"] = str(tmp_path / cache_dir)
    datamodule.config = OmegaConf.create(
        {"system": {"hardware": hardware}, "dataloader": {"multiprocessing_context": "fork"}},
    )
    readers = {"train": AttachableReader(sample_data), "valid": AttachableReader(sample_data[:4])}
    datamodule.__dict__["ds_train"] = SimpleNamespace(data_readers={"data": readers["train"]})
    datamodule.__dict__["ds_valid"] = SimpleNamespace(data_readers={"data": readers["valid"]})

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        datamodule.setup("fit")
        if cache_dir is None:
            assert datamodule.cache is None
            assert readers["train"].cache is None
            return
        cache = datamodule.cache
        assert readers["train"].cache[0] is cache
        assert readers["train"].cache[1] != readers["valid"].cache[1]  # other dates: other records
        assert len(cache.namespaces) == 2
        datamodule.teardown("fit")
        assert datamodule.cache is None
        assert cache.server is None
    finally:
        dist.destroy_process_group()
