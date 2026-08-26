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
import io
import multiprocessing
import threading
from pathlib import Path

import numpy as np
import pytest

from anemoi.training.utils.dataset_cache import (
    CacheKey,
    CacheClient,
    CacheServer,
    DatasetCache,
    DatasetCacheNamespace,
    Endpoint,
    RemoteCacheMiss,
    _is_capacity_error,
)


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
        return {
            CacheKey("dataset", 0, index): rng.standard_normal((3, 5)).astype(np.float32)
            for index in range(20)
        }

    @pytest.fixture
    def running_server(self, cache_data, tmp_path):
        """Start the production process server and yield it."""
        entries = {dataset_id: tmp_path / dataset_id for dataset_id in ("dataset", "other-dataset")}
        for key, value in cache_data.items():
            path = entries[key.dataset_id] / key.grid_id / str(key.sequence) / f"{key.position}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value)
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
        path = Path(running_server.entries[key.dataset_id]) / key.grid_id / str(key.sequence) / f"{key.position}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, expected)
        client = CacheClient(self.endpoint(running_server))
        try:
            np.testing.assert_array_equal(client.fetch(key), expected)
            np.testing.assert_array_equal(
                client.fetch(CacheKey("dataset", 0, 5)),
                cache_data[CacheKey("dataset", 0, 5)],
            )
        finally:
            client.close()

    def test_missing_entry(self, running_server):
        client = CacheClient(self.endpoint(running_server))
        try:
            with pytest.raises(RemoteCacheMiss):
                client.fetch(CacheKey("dataset", 0, 100))
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

    def test_ephemeral_port_and_clean_shutdown(self, cache_data, running_server):
        assert running_server.port > 0
        client = CacheClient(self.endpoint(running_server))
        key = CacheKey("dataset", 0, 3)
        np.testing.assert_array_equal(client.fetch(key), cache_data[key])
        client.close()

        running_server.close()
        assert not running_server.process.is_alive()


class TestDatasetCacheNamespace:
    class FakeReader:
        def __init__(self, data):
            self.data = data

        @property
        def num_sequences(self):
            return 1 if self.data.ndim == 4 else self.data.shape[0]

        def sequence_length(self, sequence=0):
            return self.data.shape[0] if self.data.ndim == 4 else self.data.shape[-2]

    def test_analysis_entry_is_committed_once(self, tmp_path, sample_data):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))

        first = sample_data[3]
        assert namespace.store(0, 3, first)
        assert not namespace.store(0, 3, np.zeros_like(first))

        np.testing.assert_array_equal(namespace.local(0, 3), sample_data[3])
        assert list(namespace.committed()) == [("all", 0, 3)]

    def test_stale_marker_is_repaired(self, tmp_path, sample_data):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))
        entry, marker, _ = namespace.paths(0, 3)
        marker.parent.mkdir(parents=True)
        marker.touch()

        assert namespace.local(0, 3) is None
        assert namespace.store(0, 3, sample_data[3])
        np.testing.assert_array_equal(namespace.local(0, 3), sample_data[3])

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
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))

        def fail_save(file, value, allow_pickle):
            file.write(b"partial")
            raise OSError("Not enough free space to write array")

        monkeypatch.setattr(np, "save", fail_save)
        with pytest.raises(OSError, match="Not enough free space"):
            namespace.store(0, 3, sample_data[3])

        entry, marker, _ = namespace.paths(0, 3)
        assert not entry.exists()
        assert not marker.exists()

    def test_negative_position(self, tmp_path, sample_data):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data))
        namespace.store(0, -1, sample_data[-1])
        np.testing.assert_array_equal(namespace.local(0, -1), sample_data[-1])


def test_check_cache_prefers_local(tmp_path, sample_data, monkeypatch):
    namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", TestDatasetCacheNamespace.FakeReader(sample_data))
    namespace.store(0, 3, sample_data[3])
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    cache._locations = {CacheKey(namespace.dataset_id, 0, 3): {1}}
    monkeypatch.setattr(cache, "_remote_cache", lambda node: pytest.fail("remote cache was checked"))

    hit, value = cache.check_cache(namespace.dataset_id, 0, [3])

    assert hit
    np.testing.assert_array_equal(value, sample_data[[3]])


def test_check_cache_returns_miss_without_reading_source(tmp_path, sample_data):
    namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", TestDatasetCacheNamespace.FakeReader(sample_data))
    cache = DatasetCache(object(), tmp_path)
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0

    assert cache.check_cache(namespace.dataset_id, 0, [3]) == (False, None)


def test_grid_shards_are_cached_separately(tmp_path, sample_data):
    namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", TestDatasetCacheNamespace.FakeReader(sample_data))
    cache = DatasetCache(object(), tmp_path)
    cache.cache_path = tmp_path
    cache.namespaces[namespace.dataset_id] = namespace
    cache.node_id = 0
    positions = [3]

    cache.store_records(namespace.dataset_id, 0, positions, sample_data[positions, ..., :2], slice(0, 2))

    hit, value = cache.check_cache(namespace.dataset_id, 0, positions, slice(0, 2))
    assert hit
    np.testing.assert_array_equal(value, sample_data[positions, ..., :2])
    assert cache.check_cache(namespace.dataset_id, 0, positions, slice(2, 4)) == (False, None)
