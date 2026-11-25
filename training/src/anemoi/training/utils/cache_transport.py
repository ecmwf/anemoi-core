# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""ZeroMQ transport for the records of the node-local dataset cache."""

import json
import logging
import multiprocessing
import os
import threading
import time
from pathlib import Path

import numpy as np
import zmq

from anemoi.training.utils import cache_format

LOGGER = logging.getLogger(__name__)

# the server writes a warning for each request that takes more time than this (seconds)
SLOW_REQUEST_S = 5.0
# the client stops to wait for a reply after this time (milliseconds)
TIMEOUT_MS = 30_000
SERVER_THREADS = 4


class RemoteCacheError(ConnectionError):
    """The cache server of another node did not reply correctly."""


def _read(cache_path: Path, request: dict) -> list[bytes]:
    """Return the compressed bytes of each requested record, or b"" for a record that is not in the cache.

    The server does not decode the records. The client decodes them. Thus, the server uses less
    CPU time, and the network moves the compressed size, not the full size of the arrays.
    """
    entries = cache_format.entries_path(cache_path, request["dataset_id"])
    return [
        cache_format.read_payload(
            cache_format.record_path(entries, request["sequence"], position, request["grid_id"]),
        )
        or b""
        for position in request["positions"]
    ]


def _worker(context: zmq.Context, endpoint: str, cache_path: Path) -> None:
    socket = context.socket(zmq.REP)
    socket.connect(endpoint)
    while True:
        message = socket.recv()
        start = time.perf_counter()
        try:
            payloads = _read(cache_path, json.loads(message))
        except Exception:
            LOGGER.exception("Dataset cache server: cannot read a request")
            payloads = [b""]  # the client sees a wrong number of records
        # one frame for each record; a record is never empty, so an empty frame means a miss
        socket.send_multipart(payloads, copy=False)
        elapsed = time.perf_counter() - start
        if elapsed > SLOW_REQUEST_S:
            LOGGER.warning(
                "Dataset cache server (pid %d): a request for %d records took %.1f s",
                os.getpid(),
                len(payloads),
                elapsed,
            )


def _serve(cache_path: Path, ready: "multiprocessing.connection.Connection") -> None:
    context = zmq.Context()
    frontend, backend = context.socket(zmq.ROUTER), context.socket(zmq.DEALER)
    try:
        port = frontend.bind_to_random_port("tcp://*")
        backend.bind("inproc://workers")
        for _ in range(min(SERVER_THREADS, os.cpu_count() or 1)):
            threading.Thread(target=_worker, args=(context, "inproc://workers", cache_path), daemon=True).start()
        ready.send((port, None))
    except Exception as error:  # noqa: BLE001 the parent process raises the error
        ready.send((0, str(error)))
        return
    finally:
        ready.close()
    zmq.proxy(frontend, backend)


class CacheClient:
    """Connection from one process to the cache server of one other node."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.socket = None

    def request(self, dataset_id: str, sequence: int, grid_id: str, positions: list[int]) -> list[np.ndarray | None]:
        """Return each requested record, or None for a record that the server does not have."""
        try:
            if self.socket is None:
                # make the socket in the process that uses it, not before a fork
                self.socket = zmq.Context.instance().socket(zmq.REQ)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)
                self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.socket.send_json(
                {"dataset_id": dataset_id, "sequence": sequence, "grid_id": grid_id, "positions": positions},
            )
            response = self.socket.recv_multipart(copy=False)
        except zmq.ZMQError as error:
            self.close()
            msg = f"{self.host}:{self.port}: {error}"
            raise RemoteCacheError(msg) from error
        if len(response) != len(positions):
            msg = f"expected {len(positions)} records from {self.host}, got {len(response)}"
            raise RemoteCacheError(msg)
        return [cache_format.decode(frame.buffer) if len(frame) else None for frame in response]

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None


class CacheServer:
    """Server process that sends the records of the cache of this node to other nodes."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = Path(cache_path)
        self.port = None
        self.process = None

    def start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_serve, args=(self.cache_path, child), daemon=True)
        self.process.start()
        child.close()
        self.port, error = parent.recv()
        parent.close()
        if error:
            raise OSError(error)

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None
