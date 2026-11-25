# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for the server of the dataset cache transport."""

import logging
import multiprocessing
import threading
import time

import pytest

from anemoi.training.utils import cache_transport
from anemoi.training.utils.cache_transport import CacheClient
from anemoi.training.utils.cache_transport import CacheKey
from anemoi.training.utils.cache_transport import Endpoint
from anemoi.training.utils.cache_transport import RemoteCacheMiss


class Ready:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def send(self, value: tuple) -> None:
        self.calls.append(("ready", value[1]))

    def close(self) -> None:
        pass


def test_server_imports_dataset_cache_before_it_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    # The import of dataset_cache can take more than the client timeout on a node with a high load.
    # Thus, the server must do it at start-up, not at the first request.
    calls = []
    monkeypatch.setattr(cache_transport, "_load_read_payload", lambda: calls.append("import"))
    monkeypatch.setattr(cache_transport.zmq, "proxy", lambda *_sockets: calls.append("proxy"))

    cache_transport._serve(0, {}, Ready(calls))

    assert calls == ["import", ("ready", None), "proxy"]


def test_server_logs_slow_requests(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(cache_transport, "SLOW_REQUEST_S", -1.0)  # every request is slow
    parent, child = multiprocessing.Pipe(duplex=False)
    threading.Thread(target=cache_transport._serve, args=(0, {}, child), daemon=True).start()
    port, error = parent.recv()
    assert error is None

    client = CacheClient(Endpoint("127.0.0.1", port, 0))

    def logged() -> bool:
        return any("a request for 1 records took" in record.getMessage() for record in caplog.records)

    with caplog.at_level(logging.WARNING, logger=cache_transport.__name__):
        with pytest.raises(RemoteCacheMiss):
            client.fetch(CacheKey("data:unknown", 0, 3, "all"))
        # the server writes the warning after it sends the reply
        deadline = time.monotonic() + 5
        while not logged() and time.monotonic() < deadline:
            time.sleep(0.01)
    client.close()

    assert logged()
