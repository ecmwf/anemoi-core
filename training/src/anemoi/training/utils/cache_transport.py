# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""ZeroMQ transport for node-local dataset cache entries."""

import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import zmq

LOGGER = logging.getLogger(__name__)

# the server writes a warning for each request that takes more time than this (seconds)
SLOW_REQUEST_S = 5.0

_READ_PAYLOAD = None


class RemoteCacheMiss(KeyError):
    pass


class RemoteCacheUnavailable(ConnectionError):
    pass


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    node_id: int


@dataclass(frozen=True)
class CacheKey:
    dataset_id: str
    sequence: int
    position: int
    grid_id: str = "all"


def _load_read_payload() -> Callable[..., bytes | None]:
    """Import dataset_cache and return read_cache_payload.

    dataset_cache imports this module at load time, thus this module cannot import dataset_cache
    at load time. The server calls this function at start-up, before it accepts requests. The
    import loads torch and pytorch_lightning, which can take a long time on a node with a high
    load (more than the 30 s timeout of the clients). If the import occurs at the first request,
    all the requests of the first remote reads wait for it and time out.
    """
    global _READ_PAYLOAD
    if _READ_PAYLOAD is None:
        from anemoi.training.utils.dataset_cache import read_cache_payload

        _READ_PAYLOAD = read_cache_payload
    return _READ_PAYLOAD


def _read(entries, request):
    """Return the compressed bytes of each requested record, or None if one is missing.

    The server does not decode the records. The client decodes them. Thus, the server uses less
    CPU time, and the network moves the compressed size, not the full size of the arrays.
    """
    read_cache_payload = _load_read_payload()

    entries_path = entries.get(request["dataset_id"])
    if entries_path is None or not request["positions"]:
        return None
    payloads = [
        read_cache_payload(entries_path, request["sequence"], position, request["grid_id"])
        for position in request["positions"]
    ]
    if any(payload is None for payload in payloads):
        return None
    return payloads


def _worker(context, endpoint, entries):
    socket = context.socket(zmq.REP)
    socket.connect(endpoint)
    while True:
        request = None
        try:
            request = socket.recv_json()
            start = time.perf_counter()
            payloads = _read(entries, request)
        except Exception:
            start = time.perf_counter()
            payloads = None
        if payloads is None:
            socket.send(b"")
        else:
            # one frame for each record; a record is never empty, so an empty frame means a miss
            socket.send_multipart(payloads, copy=False)
        elapsed = time.perf_counter() - start
        if elapsed > SLOW_REQUEST_S:
            LOGGER.warning(
                "Dataset cache server (pid %d): a request for %d records took %.1f s",
                os.getpid(),
                len(request.get("positions", [])) if isinstance(request, dict) else 0,
                elapsed,
            )


def _serve(port, entries, ready):
    context = zmq.Context()
    frontend, backend = context.socket(zmq.ROUTER), context.socket(zmq.DEALER)
    try:
        # import dataset_cache now, not at the first request (see _load_read_payload)
        _load_read_payload()
        port = frontend.bind_to_random_port("tcp://*") if port == 0 else (frontend.bind(f"tcp://*:{port}") or port)
        backend.bind("inproc://workers")
        for _ in range(min(4, os.cpu_count() or 1)):
            threading.Thread(target=_worker, args=(context, "inproc://workers", entries), daemon=True).start()
        ready.send((port, None))
    except Exception as error:
        ready.send((0, str(error)))
        return
    finally:
        ready.close()
    zmq.proxy(frontend, backend)


class CacheClient:
    def __init__(self, endpoint):
        self.endpoint, self.socket = endpoint, None

    def request_shard(self, key, positions):
        try:
            if self.socket is None:
                self.socket = zmq.Context.instance().socket(zmq.REQ)
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.setsockopt(zmq.RCVTIMEO, 30_000)
                self.socket.connect(f"tcp://{self.endpoint.host}:{self.endpoint.port}")
            self.socket.send_json(
                {
                    "dataset_id": key.dataset_id,
                    "sequence": key.sequence,
                    "positions": positions,
                    "grid_id": key.grid_id,
                },
            )
            response = self.socket.recv_multipart(copy=False)
            if len(response) == 1 and len(response[0]) == 0:
                raise RemoteCacheMiss(key)
            if len(response) != len(positions):
                msg = f"expected {len(positions)} records from {self.endpoint.host}, got {len(response)}"
                raise RemoteCacheUnavailable(msg)
            # imported lazily: dataset_cache imports this module at load time
            from anemoi.training.utils.dataset_cache import decode_cache_array

            return [decode_cache_array(frame.buffer) for frame in response]
        except RemoteCacheMiss:
            raise
        except zmq.ZMQError as error:
            self.close()
            raise RemoteCacheUnavailable(str(error)) from error

    def fetch(self, key):
        return self.request_shard(key, [key.position])[0]

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None


class CacheServer:
    def __init__(self, entries, port=0):
        self.entries, self.port = entries, port

    def start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_serve, args=(self.port, self.entries, child), daemon=True)
        self.process.start()
        child.close()
        self.port, error = parent.recv()
        parent.close()
        if error:
            raise OSError(error)

    def close(self):
        self.process.terminate()
        self.process.join()
