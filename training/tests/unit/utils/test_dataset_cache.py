# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for anemoi.training.utils.dataset_cache."""

import socket
import threading
import time

import numpy as np
import pytest
import torch

from anemoi.training.utils.dataset_cache import DatasetCache
from anemoi.training.utils.dataset_cache import TCPCacheClient
from anemoi.training.utils.dataset_cache import TCPCacheServer
from anemoi.training.utils.dataset_cache import _recv_exact

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


class FakeOriginalData:
    """Minimal stand-in for an array-like dataset (e.g. zarr array)."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def __getitem__(self, index: int) -> np.ndarray:
        return self._data[index]

    def __len__(self) -> int:
        return self._data.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape


@pytest.fixture
def sample_data() -> np.ndarray:
    """4D sample data: (time=10, channels=3, levels=2, grid=5)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 3, 2, 5)).astype(np.float32)


@pytest.fixture
def fake_original(sample_data: np.ndarray) -> FakeOriginalData:
    return FakeOriginalData(sample_data)


# ---------------------------------------------------------------------------
# Tests for _recv_exact
# ---------------------------------------------------------------------------


class TestRecvExact:
    """Tests for the _recv_exact helper function."""

    def test_receives_full_payload(self) -> None:
        """Receive exactly N bytes split across multiple recv calls."""
        payload = b"hello world! this is a test payload"
        # Create a mock socket that delivers chunks
        chunks = [payload[:5], payload[5:15], payload[15:]]

        class FakeSocket:
            def __init__(self, chunks: list[bytes]) -> None:
                self._chunks = list(chunks)

            def recv(self, nbytes: int) -> bytes:
                if not self._chunks:
                    return b""
                chunk = self._chunks.pop(0)
                return chunk[:nbytes]

        sock = FakeSocket(chunks)
        result = _recv_exact(sock, len(payload))
        assert result == payload

    def test_returns_none_on_eof(self) -> None:
        """Return None if socket is closed before all bytes arrive."""

        class FakeSocket:
            def recv(self, nbytes: int) -> bytes:  # noqa: ARG002
                return b""

        sock = FakeSocket()
        result = _recv_exact(sock, 10)
        assert result is None

    def test_partial_eof(self) -> None:
        """Return None if connection closes mid-stream."""

        class FakeSocket:
            def __init__(self):
                self._calls = 0

            def recv(self, nbytes: int) -> bytes:  # noqa: ARG002
                self._calls += 1
                if self._calls == 1:
                    return b"abc"
                return b""

        sock = FakeSocket()
        result = _recv_exact(sock, 10)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for TCPCacheServer + TCPCacheClient
# ---------------------------------------------------------------------------


class TestTCPCacheServerClient:
    """Integration tests for the TCP cache server and client."""

    @pytest.fixture
    def cache_data(self) -> dict[int, np.ndarray]:
        """Create fake cache data (dict-like indexable by int)."""
        rng = np.random.default_rng(99)
        return {i: rng.standard_normal((3, 5)).astype(np.float32) for i in range(20)}

    @pytest.fixture
    def server_port(self) -> int:
        """Find a free port for the test server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @pytest.fixture
    def running_server(self, cache_data: dict[int, np.ndarray], server_port: int) -> TCPCacheServer:
        """Start a TCPCacheServer and yield it; shut down after test."""
        server = TCPCacheServer(cache_data, server_port, host="127.0.0.1")
        server.start()
        time.sleep(0.1)  # give server thread time to bind
        yield server
        server._server_socket.close()

    def test_single_fetch(
        self,
        cache_data: dict[int, np.ndarray],
        server_port: int,
        running_server: TCPCacheServer,  # noqa: ARG002
    ) -> None:
        client = TCPCacheClient("127.0.0.1", server_port)
        try:
            result = client.fetch(5)
            np.testing.assert_array_equal(result, cache_data[5])
        finally:
            client.close()

    def test_multiple_fetches(
        self,
        cache_data: dict[int, np.ndarray],
        server_port: int,
        running_server: TCPCacheServer,  # noqa: ARG002
    ) -> None:
        client = TCPCacheClient("127.0.0.1", server_port)
        try:
            for idx in [0, 7, 13, 19]:
                result = client.fetch(idx)
                np.testing.assert_array_equal(result, cache_data[idx])
        finally:
            client.close()

    def test_getitem_interface(
        self,
        cache_data: dict[int, np.ndarray],
        server_port: int,
        running_server: TCPCacheServer,  # noqa: ARG002
    ) -> None:
        """TCPCacheClient supports dict-style indexing."""
        client = TCPCacheClient("127.0.0.1", server_port)
        try:
            result = client[10]
            np.testing.assert_array_equal(result, cache_data[10])
        finally:
            client.close()

    def test_concurrent_clients(
        self,
        cache_data: dict[int, np.ndarray],
        server_port: int,
        running_server: TCPCacheServer,  # noqa: ARG002
    ) -> None:
        """Multiple clients can connect simultaneously."""
        results = {}
        errors = []

        def worker(idx: int) -> None:
            try:
                c = TCPCacheClient("127.0.0.1", server_port)
                results[idx] = c.fetch(idx)
                c.close()
            except RuntimeError as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors in concurrent fetch: {errors}"
        for idx in range(5):
            np.testing.assert_array_equal(results[idx], cache_data[idx])


# ---------------------------------------------------------------------------
# Tests for DatasetCache.priority_reduce (static-like method)
# ---------------------------------------------------------------------------


class TestPriorityReduce:
    """Tests for the priority_reduce method of DatasetCache."""

    def _make_cache_obj(self) -> DatasetCache:
        """Create a minimal object with the priority_reduce method bound."""
        # priority_reduce is an instance method but doesn't use self beyond
        # the arguments it receives, so we can call it via the unbound version.
        return DatasetCache.__new__(DatasetCache)

    def test_prefers_local_node(self) -> None:
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

    def test_falls_back_to_remote(self) -> None:
        """If local node doesn't have it but remote does, use remote."""
        cache = self._make_cache_obj()
        node0_reg = torch.tensor([-1, -1, -1], dtype=torch.int32)
        node1_reg = torch.tensor([1, -1, 1], dtype=torch.int32)
        stacked = torch.stack([node0_reg, node1_reg])

        local_reg = node0_reg.clone()
        result = cache.priority_reduce(stacked, local_reg, node_id=0)

        expected = torch.tensor([1, -1, 1], dtype=torch.int32)
        assert torch.equal(result, expected)

    def test_all_uncached(self) -> None:
        """If nothing is cached anywhere, result is all -1."""
        cache = self._make_cache_obj()
        reg = torch.full((4,), -1, dtype=torch.int32)
        stacked = torch.stack([reg.clone(), reg.clone()])

        result = cache.priority_reduce(stacked, reg.clone(), node_id=0)
        expected = torch.full((4,), -1, dtype=torch.int32)
        assert torch.equal(result, expected)

    def test_multiple_remote_nodes(self) -> None:
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

    def _make_cache_with_registry(self, registry_values: list[int]) -> DatasetCache:
        cache = DatasetCache.__new__(DatasetCache)
        cache.cache_registry = torch.tensor(registry_values, dtype=torch.int32)
        return cache

    def test_miss(self) -> None:
        cache = self._make_cache_with_registry([-1, -1, -1])
        assert cache.check_cache(0) == []
        assert cache.check_cache(2) == []

    def test_hit(self) -> None:
        cache = self._make_cache_with_registry([0, 1, -1, 2])
        assert cache.check_cache(0) == [0]
        assert cache.check_cache(1) == [1]
        assert cache.check_cache(3) == [2]

    def test_boundary_index(self) -> None:
        cache = self._make_cache_with_registry([3, -1, -1, -1, 0])
        assert cache.check_cache(0) == [3]
        assert cache.check_cache(4) == [0]
