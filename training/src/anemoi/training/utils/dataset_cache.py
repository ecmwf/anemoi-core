# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Node-local dataset cache, with reads from the caches of the other nodes.

Each node has one cache on its local disk. All the ranks and dataloader workers of the node use it.
A record is one date of one dataset, or one grid shard of that date.

The dates of each dataset are divided into one contiguous part for each node. A node caches only the
full-grid records of its part. A worker gets a full-grid record from the cache of its node, then
from the node that owns the record, then from the dataset. A grid-shard record is cached by each node
that reads it, and is never read from other nodes.
"""

import errno
import hashlib
import json
import logging
import os
import shutil
import socket
from multiprocessing import Value
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist

from anemoi.training.utils import cache_format
from anemoi.training.utils.cache_format import FULL_GRID
from anemoi.training.utils.cache_transport import CacheClient
from anemoi.training.utils.cache_transport import CacheServer
from anemoi.training.utils.cache_transport import RemoteCacheError

LOGGER = logging.getLogger(__name__)


class CacheFullError(OSError):
    """The cache has reached its size limit, or its disk is full."""


class DatasetCacheNamespace:
    """The records of one dataset in the cache of this node."""

    def __init__(self, cache_path: Path, dataset_id: str, reader: Any) -> None:
        self.dataset_id = dataset_id
        self.num_sequences = int(reader.num_sequences)
        self.sequence_lengths = [int(reader.sequence_length(index)) for index in range(self.num_sequences)]
        self.grid_size = int(reader.grid_size)
        self.entries_path = cache_format.entries_path(cache_path, dataset_id)
        self.entries_path.mkdir(parents=True, exist_ok=True)

    def grid_id(self, grid_indices: slice | np.ndarray | None) -> str:
        """Return FULL_GRID if grid_indices select all the grid points in order, else a name for the shard."""
        if grid_indices is None:
            return FULL_GRID
        if isinstance(grid_indices, slice):
            if grid_indices.indices(self.grid_size) == (0, self.grid_size, 1):
                return FULL_GRID
            value = [grid_indices.start, grid_indices.stop, grid_indices.step]
        else:
            value = np.asarray(grid_indices).tolist()
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:16]

    def path(self, sequence: int, position: int, grid_id: str) -> Path:
        return cache_format.record_path(self.entries_path, sequence, position, grid_id)

    def local(self, sequence: int, position: int, grid_id: str) -> np.ndarray | None:
        """Return the record from the cache of this node, or None if it is not there."""
        payload = cache_format.read_payload(self.path(sequence, position, grid_id))
        return None if payload is None else cache_format.decode(payload)

    def store(self, sequence: int, position: int, value: np.ndarray, grid_id: str, min_free_bytes: int) -> bool:
        """Write one record, if it is not in the cache. Return True if this call wrote it.

        The record is written to a temporary file, then linked to its path. The link fails if
        the record exists. Thus, readers never see a partial record, and two writers of the same
        record do not interfere.
        """
        entry = self.path(sequence, position, grid_id)
        if entry.exists():
            return False
        payload = cache_format.encode(value)
        if shutil.disk_usage(self.entries_path).free - len(payload) < min_free_bytes:
            msg = f"less than {min_free_bytes / 1e9:.1f} GB would stay free on the disk"
            raise CacheFullError(msg)
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = entry.with_name(f"{entry.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            os.link(temporary, entry)
        except FileExistsError:
            return False
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT):
                raise CacheFullError(str(error)) from error
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return True


def _dataset_id(name: str, reader: Any) -> str:
    """Return a name for the dataset of a reader that is the same in all processes.

    Two readers get the same name only if they opened the same data in the same way. Then they
    can share records. The anemoi-datasets tree of the dataset gives the zarr paths and each
    operation on them, for example select, drop, start, end and frequency. Thus, two date ranges
    of one dataset, or two datasets with the same shape, get different names.

    Do not use the dates or the dataset metadata: for a cloned dataset, the dates can be invalid,
    and then the metadata changes from one process to the next.
    """
    identity = {
        "name": name,
        "tree": str(reader.data.tree()),
        "shape": tuple(int(size) for size in reader.data.shape),
        "dtype": str(reader.data.dtype),
        "variables": list(reader.variables),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{name}:{digest}"


class DatasetCache:
    """The cache of this node, and the connections to the caches of the other nodes.

    Make it before the dataloader workers start: the workers share its counters.
    """

    def __init__(self, cache_root: str | Path, hostname_suffix: str | None = None, max_fraction: float = 0.9) -> None:
        self.cache_root = Path(cache_root)
        self.hostname_suffix = hostname_suffix or ""
        self.max_fraction = max_fraction
        self.namespaces: dict[str, DatasetCacheNamespace] = {}
        self.node_id, self.num_nodes = 0, 1
        self.server = None
        self._clients: dict[tuple[int, int], CacheClient] = {}
        self.total_fetches = Value("q", 0)
        self.cache_hits_local = Value("q", 0)
        self.cache_hits_remote = Value("q", 0)
        self.cache_misses = Value("q", 0)
        self.cache_remote_errors = Value("q", 0)
        self.cache_full = Value("b", False)

    # ------------------------------------------------------------------
    # Set-up. All ranks must call these methods in the same order.
    # ------------------------------------------------------------------

    def _gather(self, value: Any) -> list:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, value, group=self.group)
        return gathered

    def setup(self) -> None:
        """Find the nodes, empty the cache of this node, and start the cache server of this node."""
        if not dist.is_initialized():
            msg = "Torch distributed must be initialized before dataset caching"
            raise RuntimeError(msg)
        self.group = dist.new_group(backend="gloo")
        self.global_rank = dist.get_rank()
        host = socket.gethostname()
        hosts = self._gather(host)
        unique_hosts = sorted(set(hosts))
        self.node_id, self.num_nodes = unique_hosts.index(host), len(unique_hosts)
        leaders = [hosts.index(name) for name in unique_hosts]  # the first rank on each node
        self.is_node_leader = self.global_rank == leaders[self.node_id]
        self.hostnames = [name + self.hostname_suffix for name in unique_hosts]
        self._check_hostnames()

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_root / f"cache-{hashlib.sha256(host.encode()).hexdigest()[:12]}"
        disk_bytes = shutil.disk_usage(self.cache_root).total
        # the cache and all other files together must not use more than max_fraction of the disk
        self.min_free_bytes = disk_bytes - int(self.max_fraction * disk_bytes)
        if self.is_node_leader:
            shutil.rmtree(self.cache_path, ignore_errors=True)
            self.cache_path.mkdir(parents=True)
            LOGGER.info(
                "Dataset cache on node %d (%s) keeps %.1f GB of %.1f GB free",
                self.node_id,
                self.cache_path,
                self.min_free_bytes / 1e9,
                disk_bytes / 1e9,
            )
            self.server = CacheServer(self.cache_path)
            self.server.start()
        ports = self._gather(self.server.port if self.is_node_leader else None)
        self.ports = [ports[rank] for rank in leaders]

    def _check_hostnames(self) -> None:
        """Make sure that each rank can find the address of the cache server on each other node.

        ZeroMQ does not show an error when it cannot find the address of a host. It tries to
        connect again and again, and each remote read waits until the timeout. Thus, an incorrect
        hostname_suffix makes the training slow, but shows only a timeout. This check stops
        the run at the start and shows the hostname that is not correct.
        """
        unresolved = []
        for node, hostname in enumerate(self.hostnames):
            if node == self.node_id:
                continue
            try:
                socket.getaddrinfo(hostname, None)
            except socket.gaierror as error:
                unresolved.append(f"{hostname} ({error})")
        # all ranks must stop together, else the other ranks wait in a collective
        failures = sorted({name for names in self._gather(unresolved) for name in names})
        if failures:
            msg = (
                f"Cannot find the address of the dataset cache server on other nodes: {', '.join(failures)}. "
                f"Make sure that system.hardware.hostname_suffix ({self.hostname_suffix!r}) is correct for "
                "this system, or set it to null."
            )
            raise RuntimeError(msg)

    def attach(self, readers: dict[str, Any]) -> None:
        """Send the sample reads of each gridded reader through the cache.

        Readers that have no ``set_cache`` method, for example the readers of observations, are not cached.
        """
        skipped = sorted(name for name, reader in readers.items() if not hasattr(reader, "set_cache"))
        if skipped:
            LOGGER.info("The dataset cache does not cache the datasets %s", skipped)
        readers = {name: reader for name, reader in readers.items() if name not in skipped}
        dataset_ids = {name: _dataset_id(name, reader) for name, reader in readers.items()}
        gathered = self._gather(sorted(dataset_ids.values()))
        if any(ids != gathered[0] for ids in gathered):
            msg = f"Ranks do not agree on the cached dataset_ids: {gathered}"
            raise RuntimeError(msg)
        for name, reader in readers.items():
            dataset_id = dataset_ids[name]
            self.namespaces[dataset_id] = DatasetCacheNamespace(self.cache_path, dataset_id, reader)
            reader.set_cache(self, dataset_id)

    # ------------------------------------------------------------------
    # Reads and writes, in the dataloader workers
    # ------------------------------------------------------------------

    def owner(self, namespace: DatasetCacheNamespace, sequence: int, position: int) -> int:
        """Return the node that caches this full-grid record."""
        if namespace.num_sequences == 1:
            index, size = position, namespace.sequence_lengths[0]
        else:  # forecast datasets: one part of the sequences (base dates) for each node
            index, size = sequence, namespace.num_sequences
        return index * self.num_nodes // size

    @staticmethod
    def _increment(counter: Value, amount: int) -> None:
        with counter.get_lock():
            counter.value += amount

    def _request(self, node: int, dataset_id: str, sequence: int, grid_id: str, positions: list[int]) -> list:
        """Return the records from the cache of another node. None means that the node does not have it."""
        key = (os.getpid(), node)  # one connection for each process: a socket must not cross a fork
        if key not in self._clients:
            self._clients[key] = CacheClient(self.hostnames[node], self.ports[node])
        try:
            return self._clients[key].request(dataset_id, sequence, grid_id, positions)
        except RemoteCacheError as error:
            self._increment(self.cache_remote_errors, 1)
            LOGGER.warning("Remote cache read from node %d failed: %s", node, error)
            return [None] * len(positions)

    def check_cache(
        self,
        dataset_id: str,
        sequence: int,
        positions: list[int],
        grid_indices: slice | np.ndarray | None = None,
    ) -> tuple[list[np.ndarray | None], list[int]]:
        """Return the records from the caches, and the indices in positions of the records that no cache has."""
        namespace = self.namespaces[dataset_id]
        grid_id = namespace.grid_id(grid_indices)
        values = [namespace.local(sequence, position, grid_id) for position in positions]
        missing = [index for index, value in enumerate(values) if value is None]
        self._increment(self.total_fetches, len(positions))
        self._increment(self.cache_hits_local, len(positions) - len(missing))

        if grid_id == FULL_GRID:
            remote = {}
            for index in missing:
                node = self.owner(namespace, sequence, positions[index])
                if node != self.node_id:
                    remote.setdefault(node, []).append(index)
            for node, indices in remote.items():
                records = self._request(node, dataset_id, sequence, grid_id, [positions[index] for index in indices])
                for index, value in zip(indices, records, strict=True):
                    values[index] = value
            unresolved = [index for index in missing if values[index] is None]
            self._increment(self.cache_hits_remote, len(missing) - len(unresolved))
            missing = unresolved

        self._increment(self.cache_misses, len(missing))
        return values, missing

    def store_records(
        self,
        dataset_id: str,
        sequence: int,
        positions: list[int],
        values: np.ndarray,
        grid_indices: slice | np.ndarray | None = None,
    ) -> None:
        """Write the records that this node caches. When the cache is full, it does not change after that."""
        if self.cache_full.value:
            return
        namespace = self.namespaces[dataset_id]
        grid_id = namespace.grid_id(grid_indices)
        try:
            for position, value in zip(positions, values, strict=True):
                if grid_id != FULL_GRID or self.owner(namespace, sequence, position) == self.node_id:
                    namespace.store(sequence, position, value, grid_id, self.min_free_bytes)
        except CacheFullError as error:
            with self.cache_full.get_lock():
                first = not self.cache_full.value
                self.cache_full.value = True
            if first:
                LOGGER.warning("Dataset cache on node %d is full, no more records are cached: %s", self.node_id, error)

    # ------------------------------------------------------------------
    # End of epoch and end of run
    # ------------------------------------------------------------------

    def log_counters(self) -> None:
        LOGGER.info(
            "Dataset cache, rank %d: %d fetches (%d local hits, %d remote hits, %d misses, %d remote errors)",
            self.global_rank,
            self.total_fetches.value,
            self.cache_hits_local.value,
            self.cache_hits_remote.value,
            self.cache_misses.value,
            self.cache_remote_errors.value,
        )

    def teardown(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        if self.server is not None:
            self.server.close()
            self.server = None
