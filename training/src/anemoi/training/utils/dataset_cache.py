"""Node-local dataset cache with cross-node reads over ZeroMQ."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import socket
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
import zmq

LOGGER = logging.getLogger(__name__)


def _is_capacity_error(error: OSError) -> bool:
    errnos = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
    messages = ("disk quota exceeded", "no space left on device", "not enough free space")
    return error.errno in errnos or any(message in str(error).lower() for message in messages)


class CacheConfigurationError(RuntimeError):
    pass


class RemoteCacheMiss(KeyError):
    pass


class RemoteCacheUnavailable(ConnectionError):
    pass


@dataclass(frozen=True)
class CacheKey:
    dataset_id: str
    sequence: int
    position: int


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    node_id: int


def _read_shard(entries, request):
    paths = [
        Path(entries[request["dataset_id"]]) / str(request["sequence"]) / f"{position}.npy"
        for position in request["positions"]
    ]
    if not paths or not all(path.exists() for path in paths):
        return None
    grid = slice(*request["grid"])
    first = np.load(paths[0], allow_pickle=False, mmap_mode="r")[..., grid]
    result = np.empty((len(paths), *first.shape), dtype=first.dtype)
    result[0] = first
    for index, path in enumerate(paths[1:], 1):
        result[index] = np.load(path, allow_pickle=False, mmap_mode="r")[..., grid]
    return result


def _serve_cache(port, entries, ready):
    context = zmq.Context()
    server = context.socket(zmq.REP)
    try:
        port = server.bind_to_random_port("tcp://*") if port == 0 else (server.bind(f"tcp://*:{port}") or port)
        ready.send((port, None))
    except Exception as error:
        ready.send((0, str(error)))
        return
    finally:
        ready.close()
    while True:
        request = server.recv_json()
        value = _read_shard(entries, request)
        if value is None:
            server.send(b"")
        else:
            metadata = json.dumps({"dtype": value.dtype.str, "shape": value.shape}, separators=(",", ":"))
            server.send_multipart([metadata.encode(), memoryview(value)], copy=False)


class ZMQCacheClient:
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.socket = None

    def _connect(self):
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 30_000)
        self.socket.connect(f"tcp://{self.endpoint.host}:{self.endpoint.port}")

    def fetch_many(self, key, positions, grid):
        request = {
            "dataset_id": key.dataset_id,
            "sequence": key.sequence,
            "positions": positions,
            "grid": [grid.start, grid.stop, grid.step],
        }
        for attempt in range(2):
            try:
                if self.socket is None:
                    self._connect()
                self.socket.send_json(request)
                response = self.socket.recv_multipart()
                if len(response) == 1:
                    raise RemoteCacheMiss(key)
                metadata = json.loads(response[0])
                return np.frombuffer(response[1], dtype=metadata["dtype"]).reshape(metadata["shape"])
            except RemoteCacheMiss:
                raise
            except zmq.ZMQError as error:
                self.close()
                if attempt:
                    raise RemoteCacheUnavailable(str(error)) from error

    def fetch(self, key):
        return self.fetch_many(key, [key.position], slice(None))[0]

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None


class ProcessZMQCacheServer:
    def __init__(self, entries, port=0):
        self.entries = entries
        self.port = port

    def start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_serve_cache, args=(self.port, self.entries, child), daemon=True)
        self.process.start()
        child.close()
        self.port, error = parent.recv()
        parent.close()
        if error:
            raise OSError(error)

    def close(self):
        self.process.terminate()
        self.process.join()


class DatasetCacheNamespace:
    """Atomic file-per-record cache for one dataset reader."""

    def __init__(self, root: Path, dataset_id: str, reader, _node_id=None):
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

    def paths(self, sequence, position):
        sequence, position = self.normalize(sequence, position)
        return (
            self.entries_path / str(sequence) / f"{position}.npy",
            self.markers_path / str(sequence) / str(position),
            self.locks_path / str(sequence) / f"{position}.lock",
        )

    @contextmanager
    def lock(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
            fcntl.flock(lock, fcntl.LOCK_UN)

    def local(self, sequence, position):
        entry, marker, _ = self.paths(sequence, position)
        return np.load(entry, allow_pickle=False, mmap_mode="r") if marker.exists() and entry.exists() else None

    def source(self, sequence, position):
        sequence, position = self.normalize(sequence, position)
        data = self.reader.data
        return np.asarray(data[position] if len(data.shape) == 4 else data[sequence][:, :, position, :])

    def store(self, sequence, position, value):
        entry, marker, lock = self.paths(sequence, position)
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
            (int(sequence.name), int(marker.name))
            for sequence in self.markers_path.iterdir()
            if sequence.is_dir()
            for marker in sequence.iterdir()
            if marker.name.isdigit()
        )

    _paths = paths
    fetch_local = local
    fetch_source = source
    committed_positions = committed


class DatasetCache(pl.LightningDataModule):
    """Datamodule proxy coordinating one SSD cache per node."""

    def __init__(self, ds, cache_root, hostname_suffix=None, proc_group=None):
        super().__init__()
        self.ds = ds
        self.cache_root = Path(cache_root)
        self.hostname_suffix = hostname_suffix or ""
        self.initial_proc_group = proc_group
        self.initialized = False
        self.namespaces = {}
        self._datasets = set()
        self._locations = {}
        self._registry_mtime = None
        self._clients = {}
        self.cache_hits_local = Value("q", 0)
        self.cache_hits_remote = Value("q", 0)
        self.cache_misses = Value("q", 0)
        self.total_fetches = Value("q", 0)
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
            raise CacheConfigurationError("Torch distributed must be initialized before dataset caching")
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
        self._registry_path = self.cache_path / ".remote-registry.json"
        if self.is_node_leader:
            shutil.rmtree(self.cache_path, ignore_errors=True)
            self.cache_path.mkdir(parents=True)
        dist.barrier(group=self.proc_group)
        self.initialized = True
        self._dataset(self.ds.ds_train)
        if self.is_node_leader:
            entries = {name: str(namespace.entries_path) for name, namespace in self.namespaces.items()}
            self.server = ProcessZMQCacheServer(entries)
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

    def set_epoch(self, epoch):
        self.ds.set_epoch(epoch)

    def _positions(self, namespace, sequence, positions):
        if isinstance(positions, slice):
            return list(range(*positions.indices(namespace.sequence_lengths[int(sequence)])))
        values = np.asarray(positions)
        return [int(values)] if values.ndim == 0 else [int(value) for value in values]

    def _endpoint(self, node):
        rank = self._leaders[node]
        return Endpoint(self.hostnames[rank], self._ports[node], node)

    def _client(self, node):
        key = (os.getpid(), node)
        if key not in self._clients:
            self._clients[key] = ZMQCacheClient(self._endpoint(node))
        return self._clients[key]

    def _local_or_source(self, namespace, sequence, position):
        key = CacheKey(namespace.dataset_id, int(sequence), position)
        self.total_fetches.value += 1
        value = namespace.local(sequence, position)
        if value is not None:
            self.cache_hits_local.value += 1
            return value
        for node in self._locations.get(key, set()) - {self.node_id}:
            try:
                value = self._client(node).fetch_many(key, [position], slice(None))[0]
                self.cache_hits_remote.value += 1
                return value
            except (RemoteCacheMiss, RemoteCacheUnavailable):
                pass
        self.cache_misses.value += 1
        value = namespace.source(sequence, position)
        if not self.cache_full.value:
            try:
                if shutil.disk_usage(self.cache_path).free < value.nbytes + (1 << 30):
                    raise OSError(errno.ENOSPC, "not enough free space")
                namespace.store(sequence, position, value)
            except OSError as error:
                if not _is_capacity_error(error):
                    raise
                self.cache_full.value = True
                LOGGER.warning("Dataset cache is full on node %s", self.node_id)
        return value

    def fetch_many(self, dataset_id, sequence, positions, grid_indices=None):
        namespace = self.namespaces[dataset_id]
        positions = self._positions(namespace, sequence, positions)
        if not positions:
            sample = namespace.source(sequence, 0)
            return np.empty((0, *sample[..., grid_indices].shape), dtype=sample.dtype)
        self._load_registry()
        keys = [CacheKey(dataset_id, int(sequence), position) for position in positions]
        nodes = set.intersection(*(self._locations.get(key, set()) - {self.node_id} for key in keys))
        if nodes and isinstance(grid_indices, slice):
            try:
                result = self._client(min(nodes)).fetch_many(keys[0], positions, grid_indices)
                self.total_fetches.value += len(positions)
                self.cache_hits_remote.value += len(positions)
                return result
            except (RemoteCacheMiss, RemoteCacheUnavailable):
                pass
        values = [self._local_or_source(namespace, sequence, position) for position in positions]
        if grid_indices is not None:
            values = [value[..., grid_indices] for value in values]
        return np.stack(values)

    def update_global_view(self):
        local = []
        if self.is_node_leader:
            for dataset_id, namespace in self.namespaces.items():
                local.extend((dataset_id, sequence, position, self.node_id) for sequence, position in namespace.committed())
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.proc_group)
        locations = {}
        for entries in gathered:
            for dataset_id, sequence, position, node in entries:
                locations.setdefault(CacheKey(dataset_id, sequence, position), set()).add(node)
        self._locations = locations
        if self.is_node_leader:
            payload = [[key.dataset_id, key.sequence, key.position, sorted(nodes)] for key, nodes in locations.items()]
            temporary = self._registry_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(temporary, self._registry_path)
        dist.barrier(group=self.proc_group)
        self._load_registry(True)

    def _load_registry(self, force=False):
        try:
            mtime = self._registry_path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if force or mtime != self._registry_mtime:
            self._locations = {
                CacheKey(dataset_id, sequence, position): set(nodes)
                for dataset_id, sequence, position, nodes in json.loads(self._registry_path.read_text())
            }
            self._registry_mtime = mtime

    def teardown(self, stage=None):
        for client in self._clients.values():
            client.close()
        if self.is_node_leader:
            self.server.close()
        self.ds.teardown(stage)

    close = teardown