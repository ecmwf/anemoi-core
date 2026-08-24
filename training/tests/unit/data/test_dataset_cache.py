# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for anemoi.training.utils.dataset_cache."""

import io
import errno
import socket
import struct
import threading
import time

import numpy as np
import pytest
import torch

from anemoi.training.utils.dataset_cache import (
    CachedDataWrapper,
    CacheKey,
    DatasetCache,
    DatasetCacheNamespace,
    Endpoint,
    RemoteCacheMiss,
    TCPCacheClient,
    TCPCacheServer,
    _recv_exact,
    _is_capacity_error,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


class FakeOriginalData:
    """Minimal stand-in for an array-like dataset (e.g. zarr array)."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return self._data.shape[0]

    @property
    def shape(self):
        return self._data.shape


class FakeCacheInstance:
    """Minimal stand-in for a DatasetCache to be used by CachedDataWrapper."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def fetch(self, date, verbose=False):
        return self._data[date]


@pytest.fixture
def sample_data():
    """4D sample data: (time=10, channels=3, levels=2, grid=5)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 3, 2, 5)).astype(np.float32)


@pytest.fixture
def fake_cache(sample_data):
    return FakeCacheInstance(sample_data)


@pytest.fixture
def fake_original(sample_data):
    return FakeOriginalData(sample_data)


# ---------------------------------------------------------------------------
# Tests for _recv_exact
# ---------------------------------------------------------------------------


class TestRecvExact:
    """Tests for the _recv_exact helper function."""

    def test_receives_full_payload(self):
        """Receive exactly N bytes split across multiple recv calls."""
        payload = b"hello world! this is a test payload"
        # Create a mock socket that delivers chunks
        chunks = [payload[:5], payload[5:15], payload[15:]]

        class FakeSocket:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            def recv(self, nbytes):
                if not self._chunks:
                    return b""
                chunk = self._chunks.pop(0)
                return chunk[:nbytes]

        sock = FakeSocket(chunks)
        result = _recv_exact(sock, len(payload))
        assert result == payload

    def test_returns_none_on_eof(self):
        """Return None if socket is closed before all bytes arrive."""

        class FakeSocket:
            def recv(self, nbytes):
                return b""

        sock = FakeSocket()
        result = _recv_exact(sock, 10)
        assert result is None

    def test_partial_eof(self):
        """Return None if connection closes mid-stream."""

        class FakeSocket:
            def __init__(self):
                self._calls = 0

            def recv(self, nbytes):
                self._calls += 1
                if self._calls == 1:
                    return b"abc"
                return b""

        sock = FakeSocket()
        result = _recv_exact(sock, 10)
        assert result is None


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
# Tests for CachedDataWrapper
# ---------------------------------------------------------------------------


class TestCachedDataWrapper:
    """Tests for the CachedDataWrapper class."""

    def test_integer_index(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[3]
        np.testing.assert_array_equal(result, sample_data[3])

    def test_numpy_integer_index(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[np.int64(7)]
        np.testing.assert_array_equal(result, sample_data[7])

    def test_simple_slice(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[2:5]
        expected = sample_data[2:5]
        np.testing.assert_array_equal(result, expected)

    def test_slice_with_step(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[0:8:2]
        expected = sample_data[0:8:2]
        np.testing.assert_array_equal(result, expected)

    def test_tuple_single_time_index(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        # Access like data[3, :, 1, :]
        result = wrapper[3, :, 1, :]
        expected = sample_data[3][:, 1, :]
        np.testing.assert_array_equal(result, expected)

    def test_tuple_slice_time_index(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        # Access like data[1:4, :, 0, :]
        result = wrapper[1:4, :, 0, :]
        # Each time step is fetched individually and then sub-indexed
        expected = np.stack([sample_data[i][:, 0, :] for i in range(1, 4)], axis=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("indices", [[1, 4, 7], np.array([1, 4, 7], dtype=np.int64)])
    def test_integer_sequence(self, indices, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[indices]
        np.testing.assert_array_equal(result, sample_data[indices])

    @pytest.mark.parametrize("indices", [[1, 4, 7], np.array([1, 4, 7], dtype=np.int64)])
    def test_tuple_integer_sequence(self, indices, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[indices, :, 1, :]
        np.testing.assert_array_equal(result, sample_data[indices, :, 1, :])

    def test_empty_integer_sequence(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        result = wrapper[np.array([], dtype=np.int64)]
        np.testing.assert_array_equal(result, sample_data[np.array([], dtype=np.int64)])

    def test_len(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        assert len(wrapper) == 10

    def test_attribute_delegation(self, sample_data, fake_cache, fake_original):
        """Attributes not on wrapper should delegate to original data."""
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        assert wrapper.shape == sample_data.shape

    def test_access_count_increments(self, sample_data, fake_cache, fake_original):
        wrapper = CachedDataWrapper(fake_original, fake_cache)
        assert wrapper._access_count == 0
        _ = wrapper[0]
        _ = wrapper[1]
        _ = wrapper[2]
        assert wrapper._access_count == 3


# ---------------------------------------------------------------------------
# Tests for TCPCacheServer + TCPCacheClient
# ---------------------------------------------------------------------------


class TestTCPCacheServerClient:
    """Integration tests for the TCP cache server and client."""

    @pytest.fixture
    def cache_data(self):
        """Create fake cached entries keyed by dataset, sequence, and position."""
        rng = np.random.default_rng(99)
        return {
            CacheKey("dataset", 0, index): rng.standard_normal((3, 5)).astype(np.float32)
            for index in range(20)
        }

    @pytest.fixture
    def server_port(self):
        """Find a free port for the test server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @pytest.fixture
    def running_server(self, cache_data, server_port):
        """Start a TCPCacheServer and yield it; shut down after test."""
        class FakeCache:
            def _fetch_local_only(self, dataset_id, sequence, position):
                return cache_data.get(CacheKey(dataset_id, sequence, position))

        server = TCPCacheServer(FakeCache(), server_port, host="127.0.0.1")
        server.start()
        time.sleep(0.1)  # give server thread time to bind
        yield server
        server.close()

    def endpoint(self, server_port):
        return Endpoint("127.0.0.1", server_port, 0)

    def test_single_fetch(self, cache_data, server_port, running_server):
        client = TCPCacheClient(self.endpoint(server_port))
        try:
            key = CacheKey("dataset", 0, 5)
            result = client.fetch(key)
            np.testing.assert_array_equal(result, cache_data[key])
        finally:
            client.close()

    def test_multiple_fetches(self, cache_data, server_port, running_server):
        client = TCPCacheClient(self.endpoint(server_port))
        try:
            for idx in [0, 7, 13, 19]:
                key = CacheKey("dataset", 0, idx)
                result = client.fetch(key)
                np.testing.assert_array_equal(result, cache_data[key])
        finally:
            client.close()

    def test_missing_entry(self, server_port, running_server):
        client = TCPCacheClient(self.endpoint(server_port))
        try:
            with pytest.raises(RemoteCacheMiss):
                client.fetch(CacheKey("dataset", 0, 100))
        finally:
            client.close()

    def test_concurrent_clients(self, cache_data, server_port, running_server):
        """Multiple clients can connect simultaneously."""
        results = {}
        errors = []

        def worker(idx):
            try:
                c = TCPCacheClient(self.endpoint(server_port))
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

    def test_ephemeral_port_and_clean_shutdown(self, cache_data):
        class FakeCache:
            def _fetch_local_only(self, dataset_id, sequence, position):
                return cache_data.get(CacheKey(dataset_id, sequence, position))

        server = TCPCacheServer(FakeCache(), 0, host="127.0.0.1")
        server.start()

        assert server.port > 0
        client = TCPCacheClient(Endpoint("127.0.0.1", server.port, 0))
        key = CacheKey("dataset", 0, 3)
        np.testing.assert_array_equal(client.fetch(key), cache_data[key])
        client.close()

        server.close()
        assert not server._thread.is_alive()


# ---------------------------------------------------------------------------
# Tests for DatasetCache.priority_reduce (static-like method)
# ---------------------------------------------------------------------------


class TestPriorityReduce:
    """Tests for the priority_reduce method of DatasetCache."""

    def _make_cache_obj(self):
        """Create a minimal object with the priority_reduce method bound."""
        # priority_reduce is an instance method but doesn't use self beyond
        # the arguments it receives, so we can call it via the unbound version.
        return DatasetCache.__new__(DatasetCache)

    def test_prefers_local_node(self):
        """If the local node has cached a date, prefer it."""
        cache = self._make_cache_obj()
        # 2 nodes (node 0 and node 1), 5 dates
        # Node 0's view: dates 0,1 cached on node 0
        # Node 1's view: dates 1,2 cached on node 1
        node0_reg = torch.tensor([0, 0, -1, -1, -1], dtype=torch.int32)
        node1_reg = torch.tensor([-1, 1, 1, -1, -1], dtype=torch.int32)
        stacked = torch.stack([node0_reg, node1_reg])

        local_reg = node0_reg.clone()
        result = cache.priority_reduce(stacked, local_reg, node_id=0)

        # date 0: only on node 0 -> 0
        # date 1: on both -> prefer local (node 0)
        # date 2: only on node 1 -> 1
        # date 3,4: nowhere -> -1
        expected = torch.tensor([0, 0, 1, -1, -1], dtype=torch.int32)
        assert torch.equal(result, expected)

    def test_falls_back_to_remote(self):
        """If local node doesn't have it but remote does, use remote."""
        cache = self._make_cache_obj()
        node0_reg = torch.tensor([-1, -1, -1], dtype=torch.int32)
        node1_reg = torch.tensor([1, -1, 1], dtype=torch.int32)
        stacked = torch.stack([node0_reg, node1_reg])

        local_reg = node0_reg.clone()
        result = cache.priority_reduce(stacked, local_reg, node_id=0)

        expected = torch.tensor([1, -1, 1], dtype=torch.int32)
        assert torch.equal(result, expected)

    def test_all_uncached(self):
        """If nothing is cached anywhere, result is all -1."""
        cache = self._make_cache_obj()
        reg = torch.full((4,), -1, dtype=torch.int32)
        stacked = torch.stack([reg.clone(), reg.clone()])

        result = cache.priority_reduce(stacked, reg.clone(), node_id=0)
        expected = torch.full((4,), -1, dtype=torch.int32)
        assert torch.equal(result, expected)

    def test_multiple_remote_nodes(self):
        """With 3 nodes, pick highest node_id as fallback (max-based)."""
        cache = self._make_cache_obj()
        # 3 nodes, 4 dates
        node0_reg = torch.tensor([-1, -1, -1, 0], dtype=torch.int32)
        node1_reg = torch.tensor([1, -1, 1, -1], dtype=torch.int32)
        node2_reg = torch.tensor([2, 2, -1, -1], dtype=torch.int32)
        stacked = torch.stack([node0_reg, node1_reg, node2_reg])

        local_reg = node0_reg.clone()
        result = cache.priority_reduce(stacked, local_reg, node_id=0)

        # date 0: node 1 and 2 have it, no local -> fallback max = 2
        # date 1: only node 2 -> 2
        # date 2: only node 1 -> 1
        # date 3: local (node 0) -> 0
        expected = torch.tensor([2, 2, 1, 0], dtype=torch.int32)
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# Tests for DatasetCache.check_cache
# ---------------------------------------------------------------------------


class TestCheckCache:
    """Tests for the check_cache method."""

    def _make_cache_with_registry(self, registry_values):
        cache = DatasetCache.__new__(DatasetCache)
        cache.cache_registry = torch.tensor(registry_values, dtype=torch.int32)
        return cache

    def test_miss(self):
        cache = self._make_cache_with_registry([-1, -1, -1])
        assert cache.check_cache(0) == []
        assert cache.check_cache(2) == []

    def test_hit(self):
        cache = self._make_cache_with_registry([0, 1, -1, 2])
        assert cache.check_cache(0) == [0]
        assert cache.check_cache(1) == [1]
        assert cache.check_cache(3) == [2]

    def test_boundary_index(self):
        cache = self._make_cache_with_registry([3, -1, -1, -1, 0])
        assert cache.check_cache(0) == [3]
        assert cache.check_cache(4) == [0]


class TestCacheWrites:
    """Tests for first-axis indexing and chunk-protected writes."""

    class FakeZarr:
        chunks = (4, 2)

        def __init__(self):
            self.data = {}
            self.write_count = 0

        def __setitem__(self, index, value):
            self.data[index] = value
            self.write_count += 1

    def _make_cache(self, tmp_path):
        cache = DatasetCache.__new__(DatasetCache)
        cache.cache = self.FakeZarr()
        cache.cache_registry = torch.full((10,), -1, dtype=torch.int32)
        cache.node_id = 2
        cache._chunk_lock_dir = tmp_path / "locks"
        cache._cache_entry_dir = tmp_path / "entries"
        cache._chunk_lock_dir.mkdir()
        cache._cache_entry_dir.mkdir()
        return cache

    def test_normalizes_negative_first_axis_index(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache._normalize_date_index(-1) == 9
        with pytest.raises(IndexError):
            cache._normalize_date_index(10)

    def test_entry_is_written_only_once(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache._write_cache_entry(3, np.array([1]))
        cache._write_cache_entry(3, np.array([2]))

        assert cache.cache.write_count == 1
        np.testing.assert_array_equal(cache.cache.data[3], np.array([1]))
        assert cache.cache_registry[3].item() == cache.node_id
        assert (cache._chunk_lock_dir / "0.lock").exists()


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
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data), 0)

        first = namespace.fetch_source(0, 3)
        assert namespace.store(0, 3, first)
        assert not namespace.store(0, 3, np.zeros_like(first))

        np.testing.assert_array_equal(namespace.fetch_local(0, 3), sample_data[3])
        assert list(namespace.committed_positions()) == [(0, 3)]

    def test_trajectory_sequences_do_not_collide(self, tmp_path):
        data = np.arange(2 * 3 * 1 * 4 * 5).reshape(2, 3, 1, 4, 5)
        namespace = DatasetCacheNamespace(tmp_path, "trajectory:fingerprint", self.FakeReader(data), 0)

        value_0 = namespace.fetch_source(0, 2)
        value_1 = namespace.fetch_source(1, 2)
        namespace.store(0, 2, value_0)
        namespace.store(1, 2, value_1)

        np.testing.assert_array_equal(namespace.fetch_local(0, 2), data[0, :, :, 2, :])
        np.testing.assert_array_equal(namespace.fetch_local(1, 2), data[1, :, :, 2, :])

    def test_failed_write_removes_partial_entry(self, tmp_path, sample_data, monkeypatch):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data), 0)

        def fail_save(file, value, allow_pickle):
            file.write(b"partial")
            raise OSError("Not enough free space to write array")

        monkeypatch.setattr(np, "save", fail_save)
        with pytest.raises(OSError, match="Not enough free space"):
            namespace.store(0, 3, sample_data[3])

        entry, marker, _ = namespace._paths(0, 3)
        assert not entry.exists()
        assert not marker.exists()
