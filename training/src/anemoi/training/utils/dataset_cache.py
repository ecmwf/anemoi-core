# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Node-local dataset cache with cross-node reads over ZeroMQ."""

import errno
import fcntl
import hashlib
import json
import logging
import os
import pickle
import shutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
from numcodecs import Blosc

from anemoi.training.utils.cache_transport import CacheClient
from anemoi.training.utils.cache_transport import CacheKey
from anemoi.training.utils.cache_transport import CacheServer
from anemoi.training.utils.cache_transport import Endpoint
from anemoi.training.utils.cache_transport import RemoteCacheMiss
from anemoi.training.utils.cache_transport import RemoteCacheUnavailable

LOGGER = logging.getLogger(__name__)

# Records are read and written inside dataloader workers, where numcodecs runs blosc
# single-threaded (it only uses threads in the main process). lz4 decodes ~2x and
# encodes ~3-8x faster than zstd there, for a similar compression ratio.
_BLOSC = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)


def load_cache_array(path):
    """Read an uncompressed .npy header followed by the blosc-compressed array data."""
    with path.open("rb") as source:
        np.lib.format.read_magic(source)
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
        payload = source.read()
    value = np.empty(shape, dtype=dtype, order="F" if fortran_order else "C")
    _BLOSC.decode(payload, out=value)
    return value


def save_cache_array(target, value):
    value = np.ascontiguousarray(value)
    np.lib.format.write_array_header_1_0(target, np.lib.format.header_data_from_array_1_0(value))
    target.write(_BLOSC.encode(value))


def read_cache_entry(entries_path, sequence, position, grid_id="all"):
    entry = Path(entries_path) / grid_id / str(sequence) / f"{position}.npy.blosc"
    return load_cache_array(entry) if entry.exists() else None


def _is_capacity_error(error: OSError) -> bool:
    errnos = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
    messages = ("disk quota exceeded", "no space left on device", "not enough free space")
    return error.errno in errnos or any(message in str(error).lower() for message in messages)


class DatasetCacheNamespace:
    """Atomic file-per-record cache for one dataset reader."""

    def __init__(self, root: Path, dataset_id: str, reader):
        self.dataset_id = dataset_id
        self.reader = reader
        self.num_sequences = int(reader.num_sequences)
        self.sequence_lengths = [int(reader.sequence_length(index)) for index in range(self.num_sequences)]
        self.path = root / hashlib.sha256(dataset_id.encode()).hexdigest()[:20]
        self.entries_path = self.path / "entries"
        self.locks_path = self.path / "locks"
        for path in (self.entries_path, self.locks_path):
            path.mkdir(parents=True, exist_ok=True)

    def normalize(self, sequence, position):
        sequence, position = int(sequence), int(position)
        if sequence < 0:
            sequence += self.num_sequences
        if not 0 <= sequence < self.num_sequences:
            raise IndexError(sequence)
        if position < 0:
            position += self.sequence_lengths[sequence]
        if not 0 <= position < self.sequence_lengths[sequence]:
            raise IndexError(position)
        return sequence, position

    def paths(self, sequence, position, grid_id="all"):
        sequence, position = self.normalize(sequence, position)
        return (
            self.entries_path / grid_id / str(sequence) / f"{position}.npy.blosc",
            self.locks_path / grid_id / str(sequence) / f"{position}.lock",
        )

    @contextmanager
    def lock(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
            fcntl.flock(lock, fcntl.LOCK_UN)

    def local(self, sequence, position, grid_id="all"):
        sequence, position = self.normalize(sequence, position)
        return read_cache_entry(self.entries_path, sequence, position, grid_id)

    def store(self, sequence, position, value, grid_id="all", sync=False):
        entry, lock = self.paths(sequence, position, grid_id)
        with self.lock(lock):
            if entry.exists():
                return False
            entry.parent.mkdir(parents=True, exist_ok=True)
            temporary = entry.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                with temporary.open("wb") as target:
                    save_cache_array(target, value)
                    if sync:
                        target.flush()
                        os.fsync(target.fileno())
                os.replace(temporary, entry)
                return True
            except OSError:
                temporary.unlink(missing_ok=True)
                raise

    def committed(self):
        return (
            (grid.name, int(sequence.name), int(entry.name.removesuffix(".npy.blosc")))
            for grid in self.entries_path.iterdir()
            if grid.is_dir()
            for sequence in grid.iterdir()
            if sequence.is_dir()
            for entry in sequence.glob("*.npy.blosc")
        )


class DatasetCache(pl.LightningDataModule):
    """Datamodule proxy coordinating one SSD cache per node."""

    def __init__(self, ds, cache_root, hostname_suffix=None, proc_group=None, async_writes=False):
        super().__init__()
        self.ds = ds  # must be set before any attribute access can reach __getattr__
        self.cache_root = Path(cache_root)
        self.hostname_suffix = hostname_suffix or ""
        self.initial_proc_group = proc_group
        self.async_writes = async_writes
        self.initialized = False
        self.namespaces, self._locations, self._connections = {}, {}, {}
        self._datasets = set()
        self.cache_hits_local = Value("q", 0)
        self.cache_hits_remote = Value("q", 0)
        self.cache_misses = Value("q", 0)
        self.total_fetches = Value("q", 0)
        self.cache_full = Value("b", False)
        self.cache_writes_skipped = Value("q", 0)
        self.cache_remote_errors = Value("q", 0)
        # update_global_view() runs in the training process, but check_cache() runs in dataloader
        # workers, which are forked once and persist across epochs. The view is therefore published
        # to a file and workers reload it whenever this shared version counter changes.
        self._view_version = Value("q", 0)
        self._view_loaded = 0
        self._view_path = self.cache_root / f"global-view-{os.getpid()}.pkl"
        self._writer = None
        self._pending_write = None

    def __getattr__(self, name):
        if name == "ds":
            raise AttributeError(name)
        return getattr(self.ds, name)

    def _dataset(self, dataset):
        if not self.initialized or id(dataset) in self._datasets:
            return dataset
        for name, reader in dataset.data_readers.items():
            identity = {
                "name": name,
                "shape": tuple(reader.data.shape),
                "source": str(getattr(reader.data, "path", type(reader.data).__name__)),
                "variables": list(reader.variables),
                "frequency": str(reader.frequency),
                "resolution": str(reader.resolution),
                "metadata": reader.metadata,
            }
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]
            dataset_id = f"{name}:{digest}"
            self.namespaces[dataset_id] = DatasetCacheNamespace(
                self.cache_path,
                dataset_id,
                reader,
            )
            reader.set_cache(self, dataset_id)
        self._datasets.add(id(dataset))
        return dataset

    @property
    def ds_train(self):
        return self._dataset(self.ds.ds_train)

    @property
    def ds_valid(self):
        return self._dataset(self.ds.ds_valid)

    @property
    def ds_test(self):
        return self._dataset(self.ds.ds_test)

    def setup(self, stage=None):
        if self.initialized:
            return
        self.ds.setup(stage)
        if not dist.is_initialized():
            raise RuntimeError("Torch distributed must be initialized before dataset caching")
        self.global_rank = dist.get_rank(self.initial_proc_group)
        self.world_size = dist.get_world_size(self.initial_proc_group)
        self.proc_group = dist.new_group(backend="gloo")
        host = socket.gethostname()
        hosts = [None] * self.world_size
        dist.all_gather_object(hosts, host, group=self.proc_group)
        unique_hosts = sorted(set(hosts))
        self.node_id = unique_hosts.index(host)
        self._leaders = {node: hosts.index(name) for node, name in enumerate(unique_hosts)}
        self.is_node_leader = self.global_rank == self._leaders[self.node_id]
        self.hostnames = [name + self.hostname_suffix for name in hosts]
        self.cache_path = self.cache_root / f"cache-{hashlib.sha256(host.encode()).hexdigest()[:12]}"
        if self.is_node_leader:
            shutil.rmtree(self.cache_path, ignore_errors=True)
            self.cache_path.mkdir(parents=True)
        dist.barrier(group=self.proc_group)
        self.initialized = True
        self._dataset(self.ds.ds_train)
        if self.is_node_leader:
            entries = {name: str(namespace.entries_path) for name, namespace in self.namespaces.items()}
            self.server = CacheServer(entries)
            self.server.start()
        ports = [None] * self.world_size
        dist.all_gather_object(ports, self.server.port if self.is_node_leader else None, group=self.proc_group)
        self._ports = {node: ports[rank] for node, rank in self._leaders.items()}

    def train_dataloader(self):
        return self.ds._get_dataloader(self.ds_train, "training")

    def val_dataloader(self):
        return self.ds._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self):
        return self.ds._get_dataloader(self.ds_test, "test")

    def _positions(self, namespace, sequence, positions):
        if isinstance(positions, slice):
            return list(range(*positions.indices(namespace.sequence_lengths[int(sequence)])))
        values = np.asarray(positions)
        return [int(values)] if values.ndim == 0 else [int(value) for value in values]

    def _grid_id(self, grid_indices):
        if grid_indices is None:
            return "all"
        if isinstance(grid_indices, slice):
            if grid_indices == slice(None):
                return "all"
            value = (grid_indices.start, grid_indices.stop, grid_indices.step)
        else:
            indices = np.asarray(grid_indices)
            value = (indices.dtype.str, indices.shape, indices.tolist())
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:16]

    def _endpoint(self, node):
        rank = self._leaders[node]
        return Endpoint(self.hostnames[rank], self._ports[node], node)

    def _remote_cache(self, node):
        key = (os.getpid(), node)
        if key not in self._connections:
            self._connections[key] = CacheClient(self._endpoint(node))
        return self._connections[key]

    @staticmethod
    def _increment(counter: Value, amount: int) -> None:
        with counter.get_lock():
            counter.value += amount

    def _publish_view(self, locations):
        """Make a new global view visible to this process and its dataloader workers."""
        temporary = self._view_path.with_suffix(".tmp")
        with temporary.open("wb") as target:
            pickle.dump(locations, target, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, self._view_path)
        with self._view_version.get_lock():
            self._view_version.value += 1
            self._view_loaded = self._view_version.value
        self._locations = locations

    def _refresh_view(self):
        """Reload the global view if the training process published a newer one."""
        version = self._view_version.value
        if version == self._view_loaded:
            return
        with self._view_path.open("rb") as source:
            self._locations = pickle.load(source)
        self._view_loaded = version

    def check_cache(self, dataset_id, sequence, positions, grid_indices=None):
        self._refresh_view()
        namespace = self.namespaces[dataset_id]
        positions = self._positions(namespace, sequence, positions)
        if not positions:
            return [], []

        grid_id = self._grid_id(grid_indices)
        keys = [CacheKey(dataset_id, int(sequence), position, grid_id) for position in positions]
        values = [namespace.local(sequence, position, grid_id) for position in positions]
        missing = [index for index, value in enumerate(values) if value is None]
        self._increment(self.total_fetches, len(positions))
        self._increment(self.cache_hits_local, len(positions) - len(missing))

        unresolved = set(missing)
        remote_batches = {}
        for index in missing:
            nodes = self._locations.get(keys[index], set()) - {self.node_id}
            if nodes:
                remote_batches.setdefault(min(nodes), []).append(index)

        for node, batch in remote_batches.items():
            try:
                remote_positions = [positions[index] for index in batch]
                connection = self._remote_cache(node)
                remote = connection.request_shard(keys[batch[0]], remote_positions)
                for index, value in zip(batch, remote):
                    values[index] = value
                    self._locations.setdefault(keys[index], set()).add(node)
                unresolved.difference_update(batch)
                self._increment(self.cache_hits_remote, len(batch))
            except (RemoteCacheMiss, RemoteCacheUnavailable) as error:
                self._increment(self.cache_remote_errors, 1)
                leader = getattr(self, "_leaders", {}).get(node)
                LOGGER.warning(
                    "Remote cache read from node %d (%s) failed: %s: %s",
                    node,
                    self.hostnames[leader] if leader is not None else "unknown host",
                    type(error).__name__,
                    error,
                )
                for index in batch:
                    self._locations[keys[index]].discard(node)

        missing = sorted(unresolved)
        self._increment(self.cache_misses, len(missing))
        return values, missing

    def _store_records(self, dataset_id, sequence, positions, values, grid_id, sync=False):
        if self.cache_full.value:
            return
        namespace = self.namespaces[dataset_id]
        for position, value in zip(positions, values):
            namespace.store(sequence, position, value, grid_id, sync=sync)

        # check if cache is full
        total, used, free = shutil.disk_usage(self.cache_root)
        if (used / total) > 0.8:
            LOGGER.warning("Dataset cache is full on node %s", self.node_id)
            self.cache_full.value = True

    def store_records(self, dataset_id, sequence, positions, values, grid_indices=None):
        namespace = self.namespaces[dataset_id]
        positions = self._positions(namespace, sequence, positions)
        grid_id = self._grid_id(grid_indices)
        if not self.async_writes:
            self._store_records(dataset_id, sequence, positions, values, grid_id)
            return

        if self._writer is None:
            self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-cache-writer")
        if self._pending_write is not None:
            if not self._pending_write.done():
                self._increment(self.cache_writes_skipped, len(positions))
                return
            self._pending_write.result()
        copied_values = [np.array(value, copy=True) for value in values]
        self._pending_write = self._writer.submit(
            self._store_records,
            dataset_id,
            sequence,
            positions,
            copied_values,
            grid_id,
            False,
        )

    def wait_for_pending_writes(self) -> None:
        if self._pending_write is not None:
            self._pending_write.result()
            self._pending_write = None

    def close_writer(self) -> None:
        if self._writer is not None:
            self.wait_for_pending_writes()
            self._writer.shutdown(wait=True)
            self._writer = None

    def update_global_view(self):
        """Update which nodes have each cached record."""
        # rank 0 on each node collects the local cache state
        local = []
        if self.is_node_leader:
            for dataset_id, namespace in self.namespaces.items():
                local.extend(
                    (dataset_id, sequence, position, grid_id, self.node_id)
                    for grid_id, sequence, position in namespace.committed()
                )
        # each process gathers the local cache state from all nodes
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.proc_group)
        locations = {}
        # each process updates its view of the global cache state
        for entries in gathered:
            for dataset_id, sequence, position, grid_id, node in entries:
                locations.setdefault(CacheKey(dataset_id, sequence, position, grid_id), set()).add(node)
        self._publish_view(locations)
        LOGGER.info(
            "Rank %d had %d fetches (%d local hits, %d remote hits, %d misses, %d remote errors)",
            self.global_rank,
            self.total_fetches.value,
            self.cache_hits_local.value,
            self.cache_hits_remote.value,
            self.cache_misses.value,
            self.cache_remote_errors.value,
        )

    def teardown(self, stage=None):
        self.close_writer()
        for connection in self._connections.values():
            connection.close()
        if self.is_node_leader:
            self.server.close()
        self._view_path.unlink(missing_ok=True)
        self.ds.teardown(stage)
