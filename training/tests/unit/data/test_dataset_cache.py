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
import multiprocessing
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from anemoi.training.utils.dataset_cache import CacheClient
from anemoi.training.utils.dataset_cache import CacheFullError
from anemoi.training.utils.dataset_cache import CacheKey
from anemoi.training.utils.dataset_cache import CacheServer
from anemoi.training.utils.dataset_cache import DatasetCache
from anemoi.training.utils.dataset_cache import DatasetCacheNamespace
from anemoi.training.utils.dataset_cache import Endpoint
from anemoi.training.utils.dataset_cache import NodeCacheBudget
from anemoi.training.utils.dataset_cache import RemoteCacheMiss
from anemoi.training.utils.dataset_cache import _is_capacity_error
from anemoi.training.utils.dataset_cache import encode_cache_array
from anemoi.training.utils.dataset_cache import save_cache_array

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """4D sample data: (time=10, channels=3, levels=2, grid=5)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 3, 2, 5)).astype(np.float32)


class TestCapacityErrors:
    @pytest.mark.parametrize(
        "error",
        [
            OSError(errno.ENOSPC, "No space left on device"),
            OSError("Not enough free space to write 1045875200 bytes after offset 128"),
            OSError("Disk quota exceeded"),
        ],
    )
    def test_recognizes_capacity_errors(self, error):
        assert _is_capacity_error(error)

    def test_does_not_hide_unrelated_io_error(self):
        assert not _is_capacity_error(OSError(errno.EACCES, "Permission denied"))


# ---------------------------------------------------------------------------
# Tests for ProcessZMQCacheServer + ZMQCacheClient
# ---------------------------------------------------------------------------


class TestProcessZMQCacheServerClient:
    """Integration tests for the ZeroMQ cache server and client."""

    @pytest.fixture
    def cache_data(self):
        """Create fake cached entries keyed by dataset, sequence, and position."""
        rng = np.random.default_rng(99)
        return {CacheKey("dataset", 0, index): rng.standard_normal((3, 5)).astype(np.float32) for index in range(20)}

    @pytest.fixture
    def running_server(self, cache_data, tmp_path):
        """Start the production process server and yield it."""
        entries = {dataset_id: tmp_path / dataset_id for dataset_id in ("dataset", "other-dataset")}
        for key, value in cache_data.items():
            path = entries[key.dataset_id] / key.grid_id / str(key.sequence) / f"{key.position}.npy.blosc"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as target:
                save_cache_array(target, value)
        context = multiprocessing.get_context("spawn")
        server = CacheServer({key: str(path) for key, path in entries.items()})
        server.start()
        yield server
        server.close()

    def endpoint(self, server):
        return Endpoint("127.0.0.1", server.port, 0)

    def test_single_fetch(self, cache_data, running_server):
        client = CacheClient(self.endpoint(running_server))
        try:
            key = CacheKey("dataset", 0, 5)
            result = client.fetch(key)
            np.testing.assert_array_equal(result, cache_data[key])
        finally:
            client.close()

    def test_multiple_fetches(self, cache_data, running_server):
        client = CacheClient(self.endpoint(running_server))
        try:
            for idx in [0, 7, 13, 19]:
                key = CacheKey("dataset", 0, idx)
                result = client.fetch(key)
                np.testing.assert_array_equal(result, cache_data[key])
        finally:
            client.close()

    def test_dataset_id_is_part_of_remote_key(self, cache_data, running_server):
        key = CacheKey("other-dataset", 0, 5)
        expected = np.full((3, 5), 17, dtype=np.float32)
        cache_data[key] = expected
        path = (
            Path(running_server.entries[key.dataset_id]) / key.grid_id / str(key.sequence) / f"{key.position}.npy.blosc"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as target:
            save_cache_array(target, expected)
        client = CacheClient(self.endpoint(running_server))
        try:
            np.testing.assert_array_equal(client.fetch(key), expected)
            np.testing.assert_array_equal(
                client.fetch(CacheKey("dataset", 0, 5)),
                cache_data[CacheKey("dataset", 0, 5)],
            )
        finally:
            client.close()

    @pytest.mark.parametrize("dataset_id", ["dataset", "unknown-dataset"])
    def test_missing_entry(self, running_server, dataset_id):
        client = CacheClient(self.endpoint(running_server))
        try:
            with pytest.raises(RemoteCacheMiss):
                client.fetch(CacheKey(dataset_id, 0, 100))
        finally:
            client.close()

    def test_concurrent_clients(self, cache_data, running_server):
        """Multiple clients can connect simultaneously."""
        results = {}
        errors = []

        def worker(idx):
            try:
                c = CacheClient(self.endpoint(running_server))
                key = CacheKey("dataset", 0, idx)
                results[idx] = c.fetch(key)
                c.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors in concurrent fetch: {errors}"
        for idx in range(5):
            np.testing.assert_array_equal(results[idx], cache_data[CacheKey("dataset", 0, idx)])

    def test_request_for_several_positions(self, cache_data, running_server):
        client = CacheClient(self.endpoint(running_server))
        try:
            result = client.request_shard(CacheKey("dataset", 0, 2), [2, 9, 4])
            assert len(result) == 3
            for value, position in zip(result, (2, 9, 4)):
                np.testing.assert_array_equal(value, cache_data[CacheKey("dataset", 0, position)])
        finally:
            client.close()

    def test_server_sends_the_compressed_file(self, cache_data, running_server):
        from anemoi.training.utils.cache_transport import _read

        key = CacheKey("dataset", 0, 5)
        request = {"dataset_id": key.dataset_id, "sequence": 0, "positions": [5], "grid_id": key.grid_id}
        (payload,) = _read(running_server.entries, request)
        assert payload == encode_cache_array(cache_data[key])

    def test_ephemeral_port_and_clean_shutdown(self, cache_data, running_server):
        assert running_server.port > 0
        client = CacheClient(self.endpoint(running_server))
        key = CacheKey("dataset", 0, 3)
        np.testing.assert_array_equal(client.fetch(key), cache_data[key])
        client.close()

        running_server.close()
        assert not running_server.process.is_alive()


def make_budget(tmp_path: Path, limit_bytes: int) -> NodeCacheBudget:
    budget = NodeCacheBudget(tmp_path / "usage", limit_bytes)
    budget.reset()
    return budget


class TestNodeCacheBudget:
    def test_reserve_and_release(self, tmp_path: Path) -> None:
        budget = make_budget(tmp_path, limit_bytes=100)

        assert budget.reserve(60)
        assert not budget.reserve(41)  # would exceed the limit: nothing is reserved
        assert budget.used() == 60
        assert budget.reserve(40)
        budget.release(30)
        assert budget.used() == 70

    def test_reserve_keeps_free_space_on_the_disk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        disk = SimpleNamespace(total=1000, used=850, free=150)
        monkeypatch.setattr("anemoi.training.utils.dataset_cache.shutil.disk_usage", lambda _path: disk)
        budget = NodeCacheBudget(tmp_path / "usage", limit_bytes=900, min_free_bytes=100)
        budget.reset()

        assert budget.reserve(50)  # leaves 100 bytes free
        disk.free = 100  # other files on the disk use the space
        assert not budget.reserve(1)  # the cache limit allows this, but the disk would get too full
        assert budget.used() == 50

    def test_budget_is_shared_between_processes(self, tmp_path: Path) -> None:
        budget = make_budget(tmp_path, limit_bytes=1000)

        def reserve(count: int) -> None:
            for _ in range(count):
                budget.reserve(7)

        context = multiprocessing.get_context("fork")
        processes = [context.Process(target=reserve, args=(50,)) for _ in range(4)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # 200 attempts of 7 bytes: no process may go past the limit, and no update may be lost
        assert budget.used() == 1000 // 7 * 7


class TestDatasetCacheNamespace:
    class FakeReader:
        def __init__(self, data):
            self.data = data

        @property
        def num_sequences(self):
            return 1 if self.data.ndim == 4 else self.data.shape[0]

        def sequence_length(self, sequence=0):
            return self.data.shape[0] if self.data.ndim == 4 else self.data.shape[-2]

    def test_analysis_entry_is_stored_once(self, tmp_path, sample_data):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))

        first = sample_data[3]
        assert namespace.store(0, 3, first)
        assert not namespace.store(0, 3, np.zeros_like(first))

        np.testing.assert_array_equal(namespace.local(0, 3), sample_data[3])
        assert list(namespace.committed()) == [("all", 0, 3)]

    def test_trajectory_sequences_do_not_collide(self, tmp_path):
        data = np.arange(2 * 3 * 1 * 4 * 5).reshape(2, 3, 1, 4, 5)
        namespace = DatasetCacheNamespace(tmp_path, "trajectory:fingerprint", self.FakeReader(data))

        value_0 = data[0, :, :, 2, :]
        value_1 = data[1, :, :, 2, :]
        namespace.store(0, 2, value_0)
        namespace.store(1, 2, value_1)

        np.testing.assert_array_equal(namespace.local(0, 2), data[0, :, :, 2, :])
        np.testing.assert_array_equal(namespace.local(1, 2), data[1, :, :, 2, :])

    def test_failed_write_removes_partial_entry(self, tmp_path, sample_data, monkeypatch):
        budget = make_budget(tmp_path, limit_bytes=10**9)
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data), budget)

        def fail_replace(_source: Path, _target: Path) -> None:
            raise OSError("Not enough free space to write array")

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.os.replace", fail_replace)
        with pytest.raises(CacheFullError, match="Not enough free space"):
            namespace.store(0, 3, sample_data[3])

        entry, _ = namespace.paths(0, 3)
        assert not entry.exists()
        assert not list(entry.parent.glob("*.tmp"))
        assert budget.used() == 0  # the reserved space is released

    def test_unrelated_write_error_is_not_cache_full(
        self,
        tmp_path: Path,
        sample_data: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))

        def fail_replace(_source: Path, _target: Path) -> None:
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.os.replace", fail_replace)
        with pytest.raises(PermissionError):
            namespace.store(0, 3, sample_data[3])

    def test_store_stops_at_the_limit(self, tmp_path: Path, sample_data: np.ndarray) -> None:
        record_bytes = len(encode_cache_array(sample_data[0]))
        budget = make_budget(tmp_path, limit_bytes=2 * record_bytes)
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data), budget)

        assert namespace.store(0, 0, sample_data[0])
        assert namespace.store(0, 1, sample_data[1])
        with pytest.raises(CacheFullError, match="cache limit"):
            namespace.store(0, 2, sample_data[2])

        assert sorted(position for _, _, position in namespace.committed()) == [0, 1]
        assert budget.used() == sum(path.stat().st_size for path in namespace.entries_path.rglob("*.npy.blosc"))

    def test_negative_position(self, tmp_path, sample_data):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))
        namespace.store(0, -1, sample_data[-1])
        np.testing.assert_array_equal(namespace.local(0, -1), sample_data[-1])

    def test_blosc_compression(self, tmp_path):
        data = np.zeros((10, 3, 2, 500), dtype=np.float32)
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(data))

        assert namespace.store(0, 3, data[3])
        np.testing.assert_array_equal(namespace.local(0, 3), data[3])

        entry, _ = namespace.paths(0, 3)
        assert entry.suffixes[-2:] == [".npy", ".blosc"]
        assert entry.stat().st_size < data[3].nbytes / 2


class TestDatasetIdentity:
    class FakeData:
        shape = (4, 2, 1, 5)
        dtype = np.dtype("float32")
        path = "/data/dataset.zarr"

    class FakeReader:
        num_sequences = 1
        variables = ("a", "b")
        resolution = "o96"

        def __init__(self, metadata):
            self.data = TestDatasetIdentity.FakeData()
            self.metadata = metadata
            self.cache = None

        def sequence_length(self, sequence=0):
            return 4

        def set_cache(self, cache, dataset_id):
            self.cache = dataset_id

    def dataset_id(self, tmp_path, metadata):
        from types import SimpleNamespace

        cache = DatasetCache(object(), tmp_path)
        cache.initialized, cache.cache_path, cache.budget = True, tmp_path, None
        reader = self.FakeReader(metadata)
        cache._dataset(SimpleNamespace(data_readers={"data": reader}))
        return reader.cache

    def test_dataset_id_does_not_use_metadata(self, tmp_path):
        # the metadata of a cloned dataset can change between processes (for example an invalid end_date)
        first = self.dataset_id(tmp_path, {"end_date": "1970-01-01T00:00:00"})
        second = self.dataset_id(tmp_path, {"end_date": "170884930401-06-20T02:29:53"})
        assert first == second

    def test_ranks_with_different_dataset_ids_are_rejected(self, tmp_path, monkeypatch):
        cache = DatasetCache(object(), tmp_path)
        cache.world_size, cache.proc_group, cache.namespaces = 2, None, {"data:aaaa": None}

        def all_gather_object(output, value, group=None):
            output[:] = [value, ["data:bbbb"]]

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.dist.all_gather_object", all_gather_object)
        with pytest.raises(RuntimeError, match="do not agree"):
            cache._check_dataset_ids()


class TestHostnameCheck:
    @staticmethod
    def all_gather_same(output: list, value: list, **_kwargs: object) -> None:
        output[:] = [value] * len(output)

    def cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known_hosts: set[str]) -> DatasetCache:
        cache = DatasetCache(object(), tmp_path, hostname_suffix="-ib")
        cache.world_size, cache.proc_group, cache.node_id = 2, None, 0
        cache._leaders = {0: 0, 1: 1}
        cache.hostnames = ["node-a-ib", "node-b-ib"]

        def getaddrinfo(host: str, *_args: object) -> list:
            if host not in known_hosts:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.socket.getaddrinfo", getaddrinfo)
        monkeypatch.setattr("anemoi.training.utils.dataset_cache.dist.all_gather_object", self.all_gather_same)
        return cache

    def test_known_hostnames_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cache(tmp_path, monkeypatch, {"node-a-ib", "node-b-ib"})._check_hostnames()

    def test_own_hostname_is_not_checked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # node-a-ib is the hostname of this node, so it is not in known_hosts
        self.cache(tmp_path, monkeypatch, {"node-b-ib"})._check_hostnames()

    def test_unknown_hostname_stops_the_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = self.cache(tmp_path, monkeypatch, set())
        with pytest.raises(RuntimeError, match=r"node-b-ib.*hostname_suffix \('-ib'\)"):
            cache._check_hostnames()

    def test_all_ranks_stop_when_one_rank_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = self.cache(tmp_path, monkeypatch, {"node-b-ib"})

        def all_gather_object(output: list, value: list, **_kwargs: object) -> None:
            output[:] = [value, ["node-a-ib (Name or service not known)"]]

        monkeypatch.setattr("anemoi.training.utils.dataset_cache.dist.all_gather_object", all_gather_object)
        with pytest.raises(RuntimeError, match="node-a-ib"):
            cache._check_hostnames()


def test_check_cache_prefers_local(tmp_path, sample_data, monkeypatch):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    namespace.store(0, 3, sample_data[3])
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    cache._locations = {CacheKey(namespace.dataset_id, 0, 3): {1}}
    monkeypatch.setattr(cache, "_remote_cache", lambda node: pytest.fail("remote cache was checked"))

    values, missing = cache.check_cache(namespace.dataset_id, 0, [3])

    assert missing == []
    np.testing.assert_array_equal(np.stack(values), sample_data[[3]])


def test_check_cache_returns_partial_values_and_missing_indices(tmp_path, sample_data):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    namespace.store(0, 3, sample_data[3])
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0

    values, missing = cache.check_cache(namespace.dataset_id, 0, [3, 4])

    np.testing.assert_array_equal(values[0], sample_data[3])
    assert values[1] is None
    assert missing == [1]


def test_check_cache_reads_missing_records_from_multiple_nodes(tmp_path, sample_data, monkeypatch):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    cache._locations = {
        CacheKey(namespace.dataset_id, 0, 3): {1},
        CacheKey(namespace.dataset_id, 0, 4): {2},
    }

    class Remote:
        def __init__(self, node):
            self.node = node

        def request_shard(self, key, positions):
            assert positions == ([3] if self.node == 1 else [4])
            return sample_data[positions]

    monkeypatch.setattr(cache, "_remote_cache", lambda node: Remote(node))

    values, missing = cache.check_cache(namespace.dataset_id, 0, [3, 4])

    assert missing == []
    np.testing.assert_array_equal(np.stack(values), sample_data[[3, 4]])


def test_check_cache_does_not_probe_without_location_snapshot(tmp_path, sample_data, monkeypatch):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    cache._leaders = {0: 0, 1: 1}

    monkeypatch.setattr(cache, "_remote_cache", lambda node: pytest.fail("unadvertised remote node was checked"))

    values, missing = cache.check_cache(namespace.dataset_id, 0, [3])

    assert values == [None]
    assert missing == [0]


def test_check_cache_falls_back_after_stale_remote_batch(tmp_path, sample_data, monkeypatch, caplog):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    cache._locations = {
        CacheKey(namespace.dataset_id, 0, 3): {1},
        CacheKey(namespace.dataset_id, 0, 4): {1},
    }

    class Remote:
        def request_shard(self, key, positions):
            raise RemoteCacheMiss(key)

    monkeypatch.setattr(cache, "_remote_cache", lambda node: Remote())

    values, missing = cache.check_cache(namespace.dataset_id, 0, [3, 4])

    assert values == [None, None]
    assert missing == [0, 1]
    assert cache._locations[CacheKey(namespace.dataset_id, 0, 3)] == set()
    assert cache._locations[CacheKey(namespace.dataset_id, 0, 4)] == set()
    assert cache.cache_remote_errors.value == 1
    assert "Remote cache read from node 1 (unknown host) failed: RemoteCacheMiss" in caplog.text


def test_check_cache_tries_only_one_advertised_node(tmp_path, sample_data, monkeypatch):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    key = CacheKey(namespace.dataset_id, 0, 3)
    cache._locations = {key: {1, 2}}
    requested = []

    class Remote:
        def request_shard(self, key, positions):
            requested.append((1, positions))
            raise RemoteCacheMiss(key)

    def remote_cache(node):
        assert node == 1
        return Remote()

    monkeypatch.setattr(cache, "_remote_cache", remote_cache)

    values, missing = cache.check_cache(namespace.dataset_id, 0, [3])

    assert values == [None]
    assert missing == [0]
    assert requested == [(1, [3])]
    assert cache._locations[key] == {2}


def test_grid_shards_are_cached_separately(tmp_path, sample_data):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path)
    cache.cache_path = tmp_path
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    positions = [3]

    cache.store_records(namespace.dataset_id, 0, positions, sample_data[positions, ..., :2], slice(0, 2))

    values, missing = cache.check_cache(namespace.dataset_id, 0, positions, slice(0, 2))
    assert missing == []
    np.testing.assert_array_equal(np.stack(values), sample_data[positions, ..., :2])
    values, missing = cache.check_cache(namespace.dataset_id, 0, positions, slice(2, 4))
    assert values == [None]
    assert missing == [0]


def test_full_cache_stays_locked(tmp_path: Path, sample_data: np.ndarray, caplog: pytest.LogCaptureFixture) -> None:
    record_bytes = len(encode_cache_array(sample_data[0]))
    budget = make_budget(tmp_path, limit_bytes=2 * record_bytes)
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
        budget,
    )
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0

    cache.store_records(namespace.dataset_id, 0, [0, 1, 2], sample_data[[0, 1, 2]])  # no exception
    assert cache.cache_full.value
    assert "Dataset cache on node 0 is full" in caplog.text

    budget.release(record_bytes)  # space is free again, but the cache does not take more records
    cache.store_records(namespace.dataset_id, 0, [3], sample_data[[3]])

    assert sorted(position for _, _, position in namespace.committed()) == [0, 1]


@pytest.mark.parametrize("max_fraction", [0, -0.1, 1.5])
def test_max_fraction_must_be_a_fraction(tmp_path: Path, max_fraction: float) -> None:
    with pytest.raises(ValueError, match="max_fraction"):
        DatasetCache(object(), tmp_path, max_fraction=max_fraction)


def test_async_writes_skip_work_while_writer_is_busy(tmp_path, sample_data, monkeypatch):
    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path, async_writes=True)
    cache.namespaces[namespace.dataset_id] = namespace
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    writes = []

    def store_records(dataset_id, sequence, positions, values, grid_id, sync):
        writes.append(positions)
        if len(writes) == 1:
            first_write_started.set()
            assert release_first_write.wait(timeout=5)

    monkeypatch.setattr(cache, "_store_records", store_records)
    cache.store_records(namespace.dataset_id, 0, [0], sample_data[[0]])
    assert first_write_started.wait(timeout=5)

    cache.store_records(namespace.dataset_id, 0, [1], sample_data[[1]])

    assert writes == [[0]]
    assert cache.cache_writes_skipped.value == 1

    release_first_write.set()
    cache.wait_for_pending_writes()
    cache.store_records(namespace.dataset_id, 0, [2], sample_data[[2]])
    cache.close_writer()

    assert writes == [[0], [2]]


def test_global_view_reaches_persistent_dataloader_workers(tmp_path, sample_data, monkeypatch):
    """Workers forked before update_global_view() must still see the new view.

    check_cache() runs in dataloader workers that persist across epochs, while the view is
    updated in the training process at the end of each epoch.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data import IterableDataset

    namespace = DatasetCacheNamespace(
        tmp_path,
        "analysis:fingerprint",
        TestDatasetCacheNamespace.FakeReader(sample_data),
    )
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0

    class Remote:
        def request_shard(self, key, positions):
            return sample_data[positions]

    monkeypatch.setattr(cache, "_remote_cache", lambda node: Remote())

    class Reads(IterableDataset):
        def __iter__(self):
            _, missing = cache.check_cache(namespace.dataset_id, 0, [3, 4])
            yield len(missing)

    loader = DataLoader(
        Reads(),
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    assert list(loader) == [2]  # epoch 0: nothing advertised yet
    cache._publish_view({CacheKey(namespace.dataset_id, 0, p): {1} for p in (3, 4)})
    assert list(loader) == [0]  # epoch 1: same worker process, served by node 1
    assert cache.cache_hits_remote.value == 2


def test_setup_limits_the_cache_to_a_fraction_of_the_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import torch.distributed as dist

    class FakeDatamodule:
        ds_train = SimpleNamespace(data_readers={})

        def setup(self, stage: str | None = None) -> None:
            pass

        def teardown(self, stage: str | None = None) -> None:
            pass

    disk = SimpleNamespace(total=1000 * 10**9, used=0, free=1000 * 10**9)
    monkeypatch.setattr("anemoi.training.utils.dataset_cache.shutil.disk_usage", lambda _path: disk)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        cache = DatasetCache(FakeDatamodule(), tmp_path / "missing" / "root", max_fraction=0.9)
        cache.setup("fit")  # creates the cache root
        try:
            assert cache.budget.limit_bytes == 900 * 10**9
            assert cache.budget.min_free_bytes == 100 * 10**9
            assert cache.budget.used() == 0
            assert cache.budget.path.parent == cache.cache_path
        finally:
            cache.teardown("fit")
    finally:
        dist.destroy_process_group()
