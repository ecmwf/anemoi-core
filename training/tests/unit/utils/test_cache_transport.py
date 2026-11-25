# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the server and client of the dataset cache transport."""

import logging
import multiprocessing
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

from anemoi.training.utils import cache_format
from anemoi.training.utils import cache_transport
from anemoi.training.utils.cache_transport import CacheClient
from anemoi.training.utils.cache_transport import CacheServer
from anemoi.training.utils.cache_transport import RemoteCacheError

DATASETS = ("dataset", "other-dataset")


@pytest.fixture
def records(tmp_path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Write records of two datasets in a cache directory."""
    rng = np.random.default_rng(99)
    values = {}
    for dataset_id in DATASETS:
        entries = cache_format.entries_path(tmp_path, dataset_id)
        for position in range(20):
            value = rng.standard_normal((3, 5)).astype(np.float32)
            path = cache_format.record_path(entries, 0, position)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(cache_format.encode(value))
            values[dataset_id, position] = value
    return values


@pytest.fixture
def server(tmp_path: Path, records: dict) -> CacheServer:  # noqa: ARG001 the server sends these records
    server = CacheServer(tmp_path)
    server.start()
    yield server
    server.close()


@pytest.fixture
def client(server: CacheServer) -> CacheClient:
    client = CacheClient("127.0.0.1", server.port)
    yield client
    client.close()


def test_request_several_positions(client: CacheClient, records: dict) -> None:
    result = client.request("dataset", 0, cache_format.FULL_GRID, [2, 9, 4])
    for value, position in zip(result, (2, 9, 4), strict=True):
        np.testing.assert_array_equal(value, records["dataset", position])


def test_dataset_id_selects_the_records(client: CacheClient, records: dict) -> None:
    (value,) = client.request("other-dataset", 0, cache_format.FULL_GRID, [5])
    np.testing.assert_array_equal(value, records["other-dataset", 5])


def test_missing_records_are_none(client: CacheClient, records: dict) -> None:
    result = client.request("dataset", 0, cache_format.FULL_GRID, [100, 3])
    assert result[0] is None
    np.testing.assert_array_equal(result[1], records["dataset", 3])
    assert client.request("unknown-dataset", 0, cache_format.FULL_GRID, [3]) == [None]
    assert client.request("dataset", 0, "shard", [3]) == [None]


def test_concurrent_clients(server: CacheServer, records: dict) -> None:
    results, errors = {}, []

    def read(position: int) -> None:
        client = CacheClient("127.0.0.1", server.port)
        try:
            (results[position],) = client.request("dataset", 0, cache_format.FULL_GRID, [position])
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        finally:
            client.close()

    threads = [threading.Thread(target=read, args=(position,)) for position in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    for position in range(5):
        np.testing.assert_array_equal(results[position], records["dataset", position])


def test_server_sends_the_compressed_file(tmp_path: Path, records: dict) -> None:
    request = {"dataset_id": "dataset", "sequence": 0, "grid_id": cache_format.FULL_GRID, "positions": [5]}
    assert cache_transport._read(tmp_path, request) == [cache_format.encode(records["dataset", 5])]


def test_server_process_does_not_import_torch() -> None:
    # the import of torch can take more than the client timeout on a node with a high load
    code = "import sys; import anemoi.training.utils.cache_transport; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603


def test_bad_request_gets_a_reply(server: CacheServer) -> None:
    # a reply with the wrong number of records: the client raises RemoteCacheError
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{server.port}")
    socket.send(b"not json")
    assert socket.recv_multipart() == [b""]
    socket.close(linger=0)


class EmptyReplySocket:
    def send_json(self, _request: dict) -> None:
        pass

    def recv_multipart(self, copy: bool = True) -> list:  # noqa: ARG002
        return []


def test_wrong_number_of_records_is_unavailable() -> None:
    client = CacheClient("127.0.0.1", 0)
    client.socket = EmptyReplySocket()
    with pytest.raises(RemoteCacheError, match="expected 1 records"):
        client.request("dataset", 0, cache_format.FULL_GRID, [3])


def test_server_that_does_not_reply_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_transport, "TIMEOUT_MS", 100)
    silent = zmq.Context.instance().socket(zmq.ROUTER)
    port = silent.bind_to_random_port("tcp://127.0.0.1")
    client = CacheClient("127.0.0.1", port)
    try:
        with pytest.raises(RemoteCacheError):
            client.request("dataset", 0, cache_format.FULL_GRID, [3])
        assert client.socket is None  # the next request uses a new socket
    finally:
        silent.close(linger=0)


def test_server_logs_slow_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(cache_transport, "SLOW_REQUEST_S", -1.0)  # each request is slow
    parent, child = multiprocessing.Pipe(duplex=False)
    threading.Thread(target=cache_transport._serve, args=(tmp_path, child), daemon=True).start()
    port, error = parent.recv()
    assert error is None
    client = CacheClient("127.0.0.1", port)

    def logged() -> bool:
        return any("a request for 1 records took" in record.getMessage() for record in caplog.records)

    with caplog.at_level(logging.WARNING, logger=cache_transport.__name__):
        assert client.request("dataset", 0, cache_format.FULL_GRID, [3]) == [None]
        # the server writes the warning after it sends the reply
        deadline = time.monotonic() + 5
        while not logged() and time.monotonic() < deadline:
            time.sleep(0.01)
    client.close()
    assert logged()


def test_close_stops_the_server(server: CacheServer) -> None:
    process = server.process
    server.close()
    assert not process.is_alive()
    server.close()  # a second close does nothing
