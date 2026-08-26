"""Node-local dataset cache with cross-node reads over ZeroMQ."""
import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import socket
import threading
from contextlib import contextmanager
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist

from anemoi.training.utils.cache_transport import CacheClient, CacheKey, CacheServer, Endpoint, RemoteCacheMiss
from anemoi.training.utils.cache_transport import RemoteCacheUnavailable
LOGGER = logging.getLogger(__name__)


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
        self.markers_path = self.path / "committed"
        self.locks_path = self.path / "locks"
        for path in (self.entries_path, self.markers_path, self.locks_path):
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
            self.entries_path / grid_id / str(sequence) / f"{position}.npy",
            self.markers_path / grid_id / str(sequence) / str(position),
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
        entry, marker, _ = self.paths(sequence, position, grid_id)
        return np.load(entry, allow_pickle=False, mmap_mode="r") if marker.exists() and entry.exists() else None

    def store(self, sequence, position, value, grid_id="all"):
        entry, marker, lock = self.paths(sequence, position, grid_id)
        with self.lock(lock):
            if marker.exists() and entry.exists():
                return False
            marker.unlink(missing_ok=True)
            entry.parent.mkdir(parents=True, exist_ok=True)
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = entry.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                with temporary.open("wb") as target:
                    np.save(target, value, allow_pickle=False)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, entry)
                marker.touch()
                return True
            except OSError:
                temporary.unlink(missing_ok=True)
                entry.unlink(missing_ok=True)
                raise

    def committed(self):
        return (
            (grid.name, int(sequence.name), int(marker.name))
            for grid in self.markers_path.iterdir()
            if grid.is_dir()
            for sequence in grid.iterdir()
            if sequence.is_dir()
            for marker in sequence.iterdir()
            if marker.name.isdigit()
        )

class DatasetCache(pl.LightningDataModule):
    """Datamodule proxy coordinating one SSD cache per node."""

    def __init__(self, ds, cache_root, hostname_suffix=None, proc_group=None):
        super().__init__()
        self.ds = ds
        self.cache_root = Path(cache_root)
        self.hostname_suffix = hostname_suffix or ""
        self.initial_proc_group = proc_group
        self.initialized = False
        self.namespaces, self._locations, self._connections = {}, {}, {}
        self._datasets = set()
        counters = (Value("q", 0) for _ in range(4))
        self.cache_hits_local, self.cache_hits_remote, self.cache_misses, self.total_fetches = counters
        self.cache_full = Value("b", False)

    def __getattr__(self, name):
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
            self.namespaces[dataset_id] = DatasetCacheNamespace(self.cache_path, dataset_id, reader)
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

    def check_cache(self, dataset_id, sequence, positions, grid_indices=None):
        namespace = self.namespaces[dataset_id]
        positions = self._positions(namespace, sequence, positions)
        if not positions:
            return False, None

        grid_id = self._grid_id(grid_indices)
        keys = [CacheKey(dataset_id, int(sequence), position, grid_id) for position in positions]
        values = [namespace.local(sequence, position, grid_id) for position in positions]
        missing = [index for index, value in enumerate(values) if value is None]
        self.total_fetches.value += len(positions)
        self.cache_hits_local.value += len(positions) - len(missing)

        if missing:
            nodes = set.intersection(*(self._locations.get(keys[index], set()) - {self.node_id} for index in missing))
            try:
                if not nodes:
                    raise RemoteCacheMiss(keys[missing[0]])
                remote_positions = [positions[index] for index in missing]
                remote = self._remote_cache(min(nodes)).request_shard(keys[missing[0]], remote_positions)
                for index, value in zip(missing, remote):
                    values[index] = value
                self.cache_hits_remote.value += len(missing)
            except (RemoteCacheMiss, RemoteCacheUnavailable):
                self.cache_misses.value += len(missing)
                return False, None

        return True, np.stack(values)

    def store_records(self, dataset_id, sequence, positions, values, grid_indices=None):
        if self.cache_full.value:
            return
        namespace = self.namespaces[dataset_id]
        positions = self._positions(namespace, sequence, positions)
        grid_id = self._grid_id(grid_indices)
        try:
            for position, value in zip(positions, values):
                if shutil.disk_usage(self.cache_path).free < value.nbytes + (1 << 30):
                    raise OSError(errno.ENOSPC, "not enough free space")
                namespace.store(sequence, position, value, grid_id)
        except OSError as error:
            if not _is_capacity_error(error):
                raise
            self.cache_full.value = True
            LOGGER.warning("Dataset cache is full on node %s", self.node_id)

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
        self._locations = locations

    def teardown(self, stage=None):
        for connection in self._connections.values():
            connection.close()
        if self.is_node_leader:
            self.server.close()
        self.ds.teardown(stage)