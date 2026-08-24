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
import struct
import threading

import numpy as np
import pytest

from anemoi.training.utils.dataset_cache import (
    CachedDataWrapper,
    CacheConfigurationError,
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
    def running_server(self, cache_data):
        """Start a TCPCacheServer and yield it; shut down after test."""
        class FakeCache:
            def _fetch_local_only(self, dataset_id, sequence, position):
                return cache_data.get(CacheKey(dataset_id, sequence, position))

        server = TCPCacheServer(FakeCache(), 0, host="127.0.0.1")
        server.start()
        yield server
        server.close()

    def endpoint(self, server):
        return Endpoint("127.0.0.1", server.port, 0)

    def test_single_fetch(self, cache_data, running_server):
        client = TCPCacheClient(self.endpoint(running_server))
        try:
            key = CacheKey("dataset", 0, 5)
            result = client.fetch(key)
            np.testing.assert_array_equal(result, cache_data[key])
        finally:
            client.close()

    def test_multiple_fetches(self, cache_data, running_server):
        client = TCPCacheClient(self.endpoint(running_server))
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
        client = TCPCacheClient(self.endpoint(running_server))
        try:
            np.testing.assert_array_equal(client.fetch(key), expected)
            np.testing.assert_array_equal(
                client.fetch(CacheKey("dataset", 0, 5)),
                cache_data[CacheKey("dataset", 0, 5)],
            )
        finally:
            client.close()

    def test_missing_entry(self, running_server):
        client = TCPCacheClient(self.endpoint(running_server))
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
                c = TCPCacheClient(self.endpoint(running_server))
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

    def test_stale_marker_is_repaired(self, tmp_path, sample_data):
        namespace = DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data), 0)
        entry, marker, _ = namespace._paths(0, 3)
        marker.parent.mkdir(parents=True)
        marker.touch()

        assert namespace.fetch_local(0, 3) is None
        assert namespace.store(0, 3, sample_data[3])
        np.testing.assert_array_equal(namespace.fetch_local(0, 3), sample_data[3])

    def test_trajectory_sequences_do_not_collide(self, tmp_path):
        data = np.arange(2 * 3 * 1 * 4 * 5).reshape(2, 3, 1, 4, 5)
        namespace = DatasetCacheNamespace(tmp_path, "trajectory:fingerprint", self.FakeReader(data), 0)

        value_0 = namespace.fetch_source(0, 2)
        value_1 = namespace.fetch_source(1, 2)
        namespace.store(0, 2, value_0)
        namespace.store(1, 2, value_1)

        np.testing.assert_array_equal(namespace.fetch_local(0, 2), data[0, :, :, 2, :])
        np.testing.assert_array_equal(namespace.fetch_local(1, 2), data[1, :, :, 2, :])

    def test_single_trajectory_sequence_uses_trajectory_layout(self, tmp_path):
        data = np.arange(1 * 3 * 1 * 4 * 5).reshape(1, 3, 1, 4, 5)
        namespace = DatasetCacheNamespace(tmp_path, "trajectory:fingerprint", self.FakeReader(data), 0)

        np.testing.assert_array_equal(namespace.fetch_source(0, 2), data[0, :, :, 2, :])

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

    def test_rejects_incompatible_existing_namespace(self, tmp_path, sample_data):
        DatasetCacheNamespace(tmp_path, "analysis:fingerprint", self.FakeReader(sample_data), 0)

        with pytest.raises(CacheConfigurationError, match="metadata mismatch"):
            DatasetCacheNamespace(
                tmp_path,
                "analysis:fingerprint",
                self.FakeReader(sample_data[:-1]),
                0,
            )
