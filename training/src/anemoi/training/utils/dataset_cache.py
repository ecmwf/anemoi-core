"""Node-local dataset cache with optional cross-node reads."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import http.server
import io
import json
import logging
import multiprocessing
import os
import shutil
import socket
import socketserver
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from multiprocessing import Value
from pathlib import Path
from time import perf_counter_ns
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

_CAPACITY_ERROR_MESSAGES = (
    "disk quota exceeded",
    "no space left on device",
    "not enough free space",
)
_REMOTE_CHUNK_BYTES = 48 << 20
_REMOTE_PIPELINE_DEPTH = 3


def _is_capacity_error(error: OSError) -> bool:
    """Return whether an I/O error means the cache cannot admit more data."""
    capacity_errnos = {errno.ENOSPC}
    if hasattr(errno, "EDQUOT"):
        capacity_errnos.add(errno.EDQUOT)
    if error.errno in capacity_errnos:
        return True
    message = str(error).lower()
    return any(fragment in message for fragment in _CAPACITY_ERROR_MESSAGES)


class CacheConfigurationError(RuntimeError):
    """Raised when the cache cannot be configured safely."""


class RemoteCacheMiss(KeyError):
    """Raised when a remote node does not contain a requested entry."""


class RemoteCacheUnavailable(ConnectionError):
    """Raised when a remote cache service cannot be reached."""


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


def _recv_exact(sock, nbytes):
    """Receive exactly ``nbytes`` from a socket, or return ``None`` on EOF."""
    parts = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _recv_into(sock, nbytes):
    payload = bytearray(nbytes)
    view = memoryview(payload)
    offset = 0
    while offset < nbytes:
        received = sock.recv_into(view[offset:])
        if not received:
            return None
        offset += received
    return payload


def _add(counter, value):
    with counter.get_lock():
        counter.value += value


def _read_shard(entries, request):
    grid = slice(*request["grid"])
    paths = [
        Path(entries[request["dataset_id"]]) / str(request["sequence"]) / f"{position}.npy"
        for position in request["positions"]
    ]
    if not paths or not all(path.exists() for path in paths):
        return None
    first = np.load(paths[0], allow_pickle=False, mmap_mode="r")[..., grid]
    result = np.empty((len(paths), *first.shape), dtype=first.dtype)
    result[0] = first
    for index, path in enumerate(paths[1:], 1):
        result[index] = np.load(path, allow_pickle=False, mmap_mode="r")[..., grid]
    return result


def _send_shard(connection, value):
    if value is None:
        connection.sendall(b"\x00" + struct.pack("!I", 0) + struct.pack("!Q", 0))
        return
    metadata = json.dumps({"dtype": value.dtype.str, "shape": value.shape}, separators=(",", ":")).encode()
    connection.sendall(b"\x01" + struct.pack("!I", len(metadata)) + struct.pack("!Q", value.nbytes) + metadata)
    connection.sendall(memoryview(value).cast("B"))


def _handle_file_requests(connection, entries, counters):
    try:
        while True:
            header = _recv_exact(connection, 4)
            if header is None:
                break
            payload = _recv_exact(connection, struct.unpack("!I", header)[0])
            if payload is None:
                break
            started = perf_counter_ns()
            value = _read_shard(entries, json.loads(payload))
            _add(counters[0], perf_counter_ns() - started)
            started = perf_counter_ns()
            _send_shard(connection, value)
            _add(counters[1], perf_counter_ns() - started)
    except (ConnectionResetError, BrokenPipeError, OSError, ValueError, KeyError):
        pass
    finally:
        connection.close()


def _serve_cache(port, entries, stop, ready, counters):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(64)
        server.settimeout(0.5)
        ready.send((server.getsockname()[1], None))
    except Exception as error:
        ready.send((0, str(error)))
        ready.close()
        return
    ready.close()
    try:
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            threading.Thread(target=_handle_file_requests, args=(connection, entries, counters), daemon=True).start()
    finally:
        server.close()


class CachedDataWrapper:
    """Compatibility wrapper routing first-axis indexing through a cache."""

    def __init__(self, original_data, cache_instance):
        self.original_data = original_data
        self.cache = cache_instance
        self._access_count = 0

    def __getitem__(self, index):
        self._access_count += 1
        if isinstance(index, tuple):
            time_index, rest = index[0], index[1:]
            if isinstance(time_index, (int, np.integer)):
                return self.cache.fetch(int(time_index))[rest]
            return self._fetch_many(time_index)[(slice(None), *rest)]
        if isinstance(index, (int, np.integer)):
            return self.cache.fetch(int(index))
        if isinstance(index, (slice, list, np.ndarray)):
            return self._fetch_many(index)
        return self.original_data[index]

    def _fetch_many(self, indices):
        if isinstance(indices, slice):
            normalized = list(range(*indices.indices(len(self.original_data))))
        else:
            values = np.asarray(indices)
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                return self.original_data[indices]
            normalized = values.tolist()
        if not normalized:
            return self.original_data[indices]
        return np.stack([self.cache.fetch(int(index)) for index in normalized])

    def __len__(self):
        return len(self.original_data)

    def __getattr__(self, name):
        return getattr(self.original_data, name)


class TCPCacheClient:
    """Persistent client for a process cache server."""

    def __init__(self, endpoint: Endpoint, timeout=30.0):
        self.endpoint = endpoint
        self.timeout = timeout
        self._sock = None

    def _connect(self):
        self._sock = socket.create_connection((self.endpoint.host, self.endpoint.port), timeout=self.timeout)

    def fetch(self, key: CacheKey):
        return self.fetch_many(key, [key.position], slice(None))[0]

    def fetch_many(self, key, positions, grid_indices):
        request = json.dumps(
            {
                "dataset_id": key.dataset_id,
                "sequence": key.sequence,
                "positions": positions,
                "grid": [grid_indices.start, grid_indices.stop, grid_indices.step],
            }
        ).encode()
        for attempt in range(2):
            try:
                if self._sock is None:
                    self._connect()
                self._sock.sendall(struct.pack("!I", len(request)) + request)
                header = _recv_exact(self._sock, 13)
                if header is None:
                    raise ConnectionError("server closed connection")
                if header[0] == 0:
                    raise RemoteCacheMiss(key)
                metadata = _recv_exact(self._sock, struct.unpack("!I", header[1:5])[0])
                payload = _recv_into(self._sock, struct.unpack("!Q", header[5:])[0])
                if payload is None:
                    raise ConnectionError("server closed connection")
                metadata = json.loads(metadata)
                return np.frombuffer(payload, dtype=metadata["dtype"]).reshape(metadata["shape"])
            except RemoteCacheMiss:
                raise
            except (ConnectionResetError, BrokenPipeError, ConnectionError, OSError) as error:
                self.close()
                if attempt:
                    raise RemoteCacheUnavailable(str(error)) from error
        raise RemoteCacheUnavailable(str(self.endpoint))

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class ProcessTCPCacheServer:
    def __init__(self, entries, port, counters):
        self.entries = entries
        self.port = port
        self.counters = counters

    def start(self):
        context = multiprocessing.get_context("spawn")
        self._stop = context.Event()
        ready_parent, ready_child = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_serve_cache,
            args=(self.port, self.entries, self._stop, ready_child, self.counters),
            daemon=True,
        )
        self._process.start()
        ready_child.close()
        self.port, error = ready_parent.recv()
        ready_parent.close()
        if error:
            raise OSError(error)

    def close(self):
        self._stop.set()
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                pass
        except OSError:
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)


class _ThreadingHTTPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class DatasetCacheNamespace:
    """Atomic file-per-record cache for one dataset reader."""

    def __init__(self, root: Path, dataset_id: str, reader, node_id: int):
        self.dataset_id = dataset_id
        self.reader = reader
        self.node_id = node_id
        self.num_sequences = int(reader.num_sequences)
        self.sequence_lengths = [int(reader.sequence_length(index)) for index in range(self.num_sequences)]
        self.token = hashlib.sha256(dataset_id.encode()).hexdigest()[:20]
        self.path = root / self.token
        self.entries_path = self.path / "entries"
        self.markers_path = self.path / "committed"
        self.locks_path = self.path / "locks"
        for directory in (self.entries_path, self.markers_path, self.locks_path):
            directory.mkdir(parents=True, exist_ok=True)
        metadata_path = self.path / "metadata.json"
        metadata = {
            "dataset_id": dataset_id,
            "num_sequences": self.num_sequences,
            "sequence_lengths": self.sequence_lengths,
            "schema_version": 1,
        }
        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) != metadata:
                raise CacheConfigurationError(f"Cache metadata mismatch for {dataset_id}")
        else:
            temporary = metadata_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(metadata, sort_keys=True))
            try:
                os.replace(temporary, metadata_path)
            except FileNotFoundError:
                pass

    def _normalize_position(self, sequence, position):
        sequence = int(sequence)
        if sequence < 0:
            sequence += self.num_sequences
        if sequence < 0 or sequence >= self.num_sequences:
            raise IndexError(f"sequence {sequence} is out of range")
        position = int(position)
        length = self.sequence_lengths[sequence]
        if position < 0:
            position += length
        if position < 0 or position >= length:
            raise IndexError(f"position {position} is out of range for sequence {sequence}")
        return sequence, position

    def _paths(self, sequence, position):
        sequence, position = self._normalize_position(sequence, position)
        entry = self.entries_path / str(sequence) / f"{position}.npy"
        marker = self.markers_path / str(sequence) / str(position)
        lock = self.locks_path / str(sequence) / f"{position}.lock"
        return entry, marker, lock

    @contextmanager
    def _entry_lock(self, lock_path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def fetch_local(self, sequence, position):
        entry, marker, _ = self._paths(sequence, position)
        if not marker.exists():
            return None
        try:
            return np.load(entry, allow_pickle=False, mmap_mode="r")
        except FileNotFoundError:
            return None

    def fetch_source(self, sequence, position):
        sequence, position = self._normalize_position(sequence, position)
        if len(self.reader.data.shape) == 4:
            return np.asarray(self.reader.data[position])
        return np.asarray(self.reader.data[sequence][:, :, position, :])

    def store(self, sequence, position, value):
        entry, marker, lock = self._paths(sequence, position)
        with self._entry_lock(lock):
            if marker.exists() and entry.exists():
                return False
            marker.unlink(missing_ok=True)
            entry.parent.mkdir(parents=True, exist_ok=True)
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = entry.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                with temporary.open("wb") as temporary_file:
                    np.save(temporary_file, np.asarray(value), allow_pickle=False)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(temporary, entry)
                marker.touch(exist_ok=True)
                return True
            except OSError:
                temporary.unlink(missing_ok=True)
                if not marker.exists():
                    entry.unlink(missing_ok=True)
                raise

    def committed_positions(self):
        for sequence_path in self.markers_path.iterdir():
            if sequence_path.is_dir():
                for marker in sequence_path.iterdir():
                    if marker.name.isdigit():
                        yield int(sequence_path.name), int(marker.name)


class DatasetCache(pl.LightningDataModule):
    """Datamodule proxy coordinating one cache per node for every data reader."""

    def __init__(
        self,
        ds,
        cache_root,
        dataset_path=None,
        proc_group=None,
        hostname_suffix=None,
        remote_backend="tcp",
        num_gpus_per_node=None,
        root_port=None,
        delete_existing_cache=True,
        min_free_bytes=1 << 30,
    ):
        super().__init__()
        self.ds = ds
        self.cache_root = Path(cache_root or os.environ.get("TMPDIR", "/tmp"))
        self.dataset_path = dataset_path
        self.initial_proc_group = proc_group
        self.hostname_suffix = hostname_suffix
        self.remote_backend = remote_backend
        self.num_gpus_per_node = int(num_gpus_per_node or 1)
        self.root_port = None if root_port is None else int(root_port)
        self.delete_existing_cache = delete_existing_cache
        self.min_free_bytes = int(min_free_bytes)
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes must not be negative")
        self.is_initalised = False
        self.namespaces = {}
        self._dataset_aliases = {}
        self._remote_locations = {}
        self._remote_registry_mtime = None
        self._tcp_clients = {}
        self._cached_datasets = {}
        self.cache_hits_local = Value("q", 0)
        self.cache_hits_remote = Value("q", 0)
        self.cache_misses = Value("q", 0)
        self.total_fetches = Value("q", 0)
        self.cache_full = Value("b", False)
        self.profile_io = os.environ.get("ANEMOI_DATASET_CACHE_PROFILE") == "1"
        self.profile_local_ns = Value("q", 0)
        self.profile_remote_ns = Value("q", 0)
        self.profile_source_ns = Value("q", 0)
        self.profile_store_ns = Value("q", 0)
        self.profile_fetch_many_ns = Value("q", 0)
        context = multiprocessing.get_context("spawn")
        self.remote_server_read_ns = context.Value("q", 0)
        self.remote_server_send_ns = context.Value("q", 0)

    def __getattr__(self, name):
        return getattr(self.ds, name)

    @property
    def ds_train(self):
        return self._attach_dataset("training", self.ds.ds_train)

    @property
    def ds_valid(self):
        return self._attach_dataset("validation", self.ds.ds_valid)

    @property
    def ds_test(self):
        return self._attach_dataset("test", self.ds.ds_test)

    def _reader_fingerprint(self, stage, name, reader):
        dates = getattr(reader.data, "dates", getattr(reader.data, "base_dates", None))
        geometry = {
            "schema_version": 1,
            "name": name,
            "shape": tuple(int(value) for value in reader.data.shape),
            "num_sequences": int(reader.num_sequences),
            "sequence_lengths": [int(reader.sequence_length(i)) for i in range(reader.num_sequences)],
            "source": str(getattr(reader.data, "path", getattr(reader.data, "name", type(reader.data).__name__))),
            "first_date": None if dates is None or len(dates) == 0 else str(dates[0]),
            "last_date": None if dates is None or len(dates) == 0 else str(dates[-1]),
            "variables": list(reader.variables),
            "frequency": str(reader.frequency),
            "resolution": str(reader.resolution),
            "metadata": reader.metadata,
        }
        digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return f"{name}:{digest}"

    def _attach_dataset(self, stage, dataset):
        if not self.is_initalised:
            return dataset
        cache_key = (stage, id(dataset))
        if cache_key in self._cached_datasets:
            return dataset
        for name, reader in dataset.data_readers.items():
            dataset_id = self._reader_fingerprint(stage, name, reader)
            namespace = DatasetCacheNamespace(self.cache_path, dataset_id, reader, self.node_id)
            self.namespaces[dataset_id] = namespace
            self._dataset_aliases.setdefault(name, dataset_id)
            reader.set_cache(self, dataset_id)
        self._cached_datasets[cache_key] = dataset
        return dataset

    def setup(self, stage=None):
        if self.is_initalised:
            return
        self.ds.setup(stage)
        if not dist.is_initialized():
            raise CacheConfigurationError("Torch distributed must be initialized before dataset caching")
        self.global_rank = dist.get_rank(self.initial_proc_group)
        self.world_size = dist.get_world_size(self.initial_proc_group)
        self.proc_group = dist.new_group(ranks=None, backend="gloo")
        host = socket.gethostname()
        raw_hostnames = [None] * self.world_size
        dist.all_gather_object(raw_hostnames, host, group=self.proc_group)
        unique_hosts = sorted(set(raw_hostnames))
        host_to_node = {value: index for index, value in enumerate(unique_hosts)}
        self.node_id = host_to_node[host]
        self.num_nodes = len(unique_hosts)
        self._node_leader_rank = {
            node_id: next(rank for rank, value in enumerate(raw_hostnames) if value == node_host)
            for node_id, node_host in enumerate(unique_hosts)
        }
        self.is_node_leader = self.global_rank == self._node_leader_rank[self.node_id]
        self._raw_hostnames = raw_hostnames
        self.hostnames = [value + (self.hostname_suffix or "") for value in raw_hostnames]
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.num_gpus_per_node))

        self.cache_dir_name = f"cache-{hashlib.sha256(host.encode()).hexdigest()[:12]}"
        self.cache_path = self.cache_root / self.cache_dir_name
        self._admission_disabled_path = self.cache_path / ".admission-disabled"
        self._remote_registry_path = self.cache_path / ".remote-registry.json"
        if self.is_node_leader:
            if self.delete_existing_cache and self.cache_path.exists():
                shutil.rmtree(self.cache_path)
            self.cache_path.mkdir(parents=True, exist_ok=True)
        dist.barrier(group=self.proc_group)
        self.is_initalised = True
        self._attach_dataset("training", self.ds.ds_train)

        self.port = 0 if self.root_port is None else self.root_port + self.node_id
        if self.is_node_leader:
            self._start_server()
        advertised_port = self.port if self.is_node_leader else None
        rank_ports = [None] * self.world_size
        dist.all_gather_object(rank_ports, advertised_port, group=self.proc_group)
        self._node_ports = {
            node_id: int(rank_ports[leader_rank])
            for node_id, leader_rank in self._node_leader_rank.items()
        }
        self.port = self._node_ports[self.node_id]
        LOGGER.info("Node %s dataset cache listening on port %s", self.node_id, self.port)

    def train_dataloader(self):
        return self.ds._get_dataloader(self.ds_train, "training")

    def val_dataloader(self):
        return self.ds._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self):
        return self.ds._get_dataloader(self.ds_test, "test")

    def set_epoch(self, epoch):
        self.ds.set_epoch(epoch)

    def _start_server(self):
        if self.remote_backend == "tcp":
            entries = {dataset_id: str(namespace.entries_path) for dataset_id, namespace in self.namespaces.items()}
            self._server = ProcessTCPCacheServer(
                entries,
                self.port,
                (self.remote_server_read_ns, self.remote_server_send_ns),
            )
            try:
                self._server.start()
            except OSError as error:
                if error.errno != errno.EADDRINUSE or self.port == 0:
                    raise
                LOGGER.warning("Dataset cache port %s is occupied; selecting a free port", self.port)
                self._server = ProcessTCPCacheServer(
                    entries,
                    0,
                    (self.remote_server_read_ns, self.remote_server_send_ns),
                )
                self._server.start()
            self.port = self._server.port
            return
        if self.remote_backend not in {"zarr", "http"}:
            raise CacheConfigurationError(f"Unknown remote cache backend: {self.remote_backend}")
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(self.cache_root))
        try:
            self._server = _ThreadingHTTPServer(("", self.port), handler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE or self.port == 0:
                raise
            LOGGER.warning("Dataset cache port %s is occupied; selecting a free port", self.port)
            self._server = _ThreadingHTTPServer(("", 0), handler)
        self.port = self._server.server_address[1]
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

    def _resolve_dataset_id(self, dataset_id):
        if dataset_id in self.namespaces:
            return dataset_id
        if dataset_id in self._dataset_aliases:
            return self._dataset_aliases[dataset_id]
        raise KeyError(f"Unknown cache dataset {dataset_id!r}")

    def _normalize_positions(self, namespace, sequence, positions):
        if isinstance(positions, slice):
            return list(range(*positions.indices(namespace.sequence_lengths[int(sequence)])))
        values = np.asarray(positions)
        if values.ndim == 0:
            return [int(values)]
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise IndexError("cache positions must be an integer, slice, or one-dimensional integer sequence")
        return [int(value) for value in values]

    def fetch(self, dataset_id, sequence=None, position=None, verbose=False):
        if position is None:
            position = dataset_id
            dataset_id = next(iter(self.namespaces))
            sequence = 0
        dataset_id = self._resolve_dataset_id(dataset_id)
        namespace = self.namespaces[dataset_id]
        sequence, position = namespace._normalize_position(sequence, position)
        key = CacheKey(dataset_id, sequence, position)
        self.total_fetches.value += 1
        started = perf_counter_ns() if self.profile_io else 0
        local = namespace.fetch_local(sequence, position)
        if started:
            self.profile_local_ns.value += perf_counter_ns() - started
        if local is not None:
            self.cache_hits_local.value += 1
            return local

        self._load_remote_registry()
        for remote_node in sorted(self._remote_locations.get(key, set()) - {self.node_id}):
            try:
                started = perf_counter_ns() if self.profile_io else 0
                value = self._fetch_remote(remote_node, key)
                if started:
                    self.profile_remote_ns.value += perf_counter_ns() - started
                self.cache_hits_remote.value += 1
                return value
            except (RemoteCacheMiss, RemoteCacheUnavailable):
                if started:
                    self.profile_remote_ns.value += perf_counter_ns() - started
                self._remote_locations.get(key, set()).discard(remote_node)

        self.cache_misses.value += 1
        started = perf_counter_ns() if self.profile_io else 0
        value = namespace.fetch_source(sequence, position)
        if started:
            self.profile_source_ns.value += perf_counter_ns() - started
        if not self.cache_full.value and not self._admission_disabled_path.exists():
            started = perf_counter_ns() if self.profile_io else 0
            required_bytes = int(np.asarray(value).nbytes) + self.min_free_bytes
            if shutil.disk_usage(self.cache_path).free < required_bytes:
                self._disable_admission(
                    f"insufficient free space for {np.asarray(value).nbytes} byte entry "
                    f"with {self.min_free_bytes} byte reserve"
                )
            else:
                try:
                    namespace.store(sequence, position, value)
                except OSError as error:
                    if not _is_capacity_error(error):
                        raise
                    self._disable_admission(str(error))
            if started:
                self.profile_store_ns.value += perf_counter_ns() - started
        return value

    def profile_snapshot(self):
        return tuple(
            counter.value
            for counter in (
                self.profile_local_ns,
                self.profile_remote_ns,
                self.profile_source_ns,
                self.profile_store_ns,
                self.profile_fetch_many_ns,
            )
        )

    def _disable_admission(self, reason):
        """Disable new writes on this node while preserving cache reads."""
        was_enabled = not self.cache_full.value and not self._admission_disabled_path.exists()
        self.cache_full.value = True
        try:
            self._admission_disabled_path.touch(exist_ok=True)
        except OSError as error:
            if not _is_capacity_error(error):
                raise
        if was_enabled:
            LOGGER.warning("Node %s dataset cache admission disabled: %s", self.node_id, reason)

    def fetch_many(self, dataset_id, sequence, positions, grid_indices=None):
        started = perf_counter_ns() if self.profile_io else 0
        dataset_id = self._resolve_dataset_id(dataset_id)
        namespace = self.namespaces[dataset_id]
        normalized = self._normalize_positions(namespace, sequence, positions)
        if not normalized:
            if namespace.num_sequences == 1:
                result = np.asarray(namespace.reader.data[np.array([], dtype=np.int64)])
            else:
                sample = namespace.fetch_source(sequence, 0)
                result = np.empty((0, *sample.shape), dtype=sample.dtype)
            return result if grid_indices is None else result[..., grid_indices]
        self._load_remote_registry()
        keys = [CacheKey(dataset_id, int(sequence), position) for position in normalized]
        remote_nodes = set.intersection(
            *(self._remote_locations.get(key, set()) - {self.node_id} for key in keys)
        )
        if remote_nodes and isinstance(grid_indices, slice):
            try:
                result = self._fetch_remote_many(min(remote_nodes), keys[0], normalized, grid_indices)
            except (RemoteCacheMiss, RemoteCacheUnavailable):
                pass
            else:
                self.total_fetches.value += len(normalized)
                self.cache_hits_remote.value += len(normalized)
                if started:
                    elapsed = perf_counter_ns() - started
                    self.profile_remote_ns.value += elapsed
                    self.profile_fetch_many_ns.value += elapsed
                return result
        values = [self.fetch(dataset_id, sequence, position) for position in normalized]
        if grid_indices is not None:
            values = [value[..., grid_indices] for value in values]
        result = np.stack(values)
        if started:
            self.profile_fetch_many_ns.value += perf_counter_ns() - started
        return result

    def _fetch_remote_many(self, node_id, key, positions, grid_indices):
        namespace = self.namespaces[key.dataset_id]
        reference = namespace.fetch_local(key.sequence, positions[0])
        if reference is None:
            reference = namespace.fetch_source(key.sequence, positions[0])
        start, stop, step = grid_indices.indices(reference.shape[-1])
        if step != 1:
            raise RemoteCacheUnavailable("remote grid shard must be contiguous")
        bytes_per_gridpoint = len(positions) * reference[..., :1].nbytes
        chunk_size = max(1, _REMOTE_CHUNK_BYTES // bytes_per_gridpoint)
        chunks = [slice(offset, min(offset + chunk_size, stop)) for offset in range(start, stop, chunk_size)]
        executor = getattr(self, "_remote_executor", None)
        if executor is None:
            executor = self._remote_executor = ThreadPoolExecutor(max_workers=_REMOTE_PIPELINE_DEPTH)
        indexed = list(enumerate(chunks))

        def fetch_lane(slot):
            client_key = (os.getpid(), node_id, slot)
            client = self._tcp_clients.get(client_key)
            if client is None:
                client = self._tcp_clients[client_key] = TCPCacheClient(self._endpoint(node_id))
            return [
                (index, client.fetch_many(key, positions, chunk))
                for index, chunk in indexed[slot::_REMOTE_PIPELINE_DEPTH]
            ]

        futures = [executor.submit(fetch_lane, slot) for slot in range(min(_REMOTE_PIPELINE_DEPTH, len(chunks)))]
        values = [item for future in futures for item in future.result()]
        values.sort(key=lambda item: item[0])
        arrays = [array for _, array in values]
        result = np.empty((*arrays[0].shape[:-1], stop - start), dtype=arrays[0].dtype)
        offset = 0
        for array in arrays:
            result[..., offset : offset + array.shape[-1]] = array
            offset += array.shape[-1]
        return result

    def _fetch_local_only(self, dataset_id, sequence, position):
        namespace = self.namespaces.get(dataset_id)
        return None if namespace is None else namespace.fetch_local(sequence, position)

    def _endpoint(self, node_id):
        rank = self._node_leader_rank[node_id]
        return Endpoint(self.hostnames[rank], self._node_ports[node_id], node_id)

    def _fetch_remote(self, node_id, key):
        endpoint = self._endpoint(node_id)
        if self.remote_backend == "tcp":
            client_key = (os.getpid(), node_id)
            client = self._tcp_clients.get(client_key)
            if client is None:
                client = self._tcp_clients[client_key] = TCPCacheClient(endpoint)
            return client.fetch(key)
        namespace = self.namespaces[key.dataset_id]
        remote_host = self._raw_hostnames[self._node_leader_rank[node_id]]
        remote_cache_dir = f"cache-{hashlib.sha256(remote_host.encode()).hexdigest()[:12]}"
        url = (
            f"http://{endpoint.host}:{endpoint.port}/{remote_cache_dir}/{namespace.token}/entries/"
            f"{key.sequence}/{key.position}.npy"
        )
        try:
            with urlopen(url, timeout=30) as response:
                return np.load(io.BytesIO(response.read()), allow_pickle=False)
        except HTTPError as error:
            if error.code == 404:
                raise RemoteCacheMiss(key) from error
            raise RemoteCacheUnavailable(str(error)) from error
        except (URLError, OSError) as error:
            raise RemoteCacheUnavailable(str(error)) from error

    def update_global_view(self):
        local_entries = []
        if self.is_node_leader:
            for dataset_id, namespace in self.namespaces.items():
                local_entries.extend(
                    (dataset_id, sequence, position, self.node_id)
                    for sequence, position in namespace.committed_positions()
                )
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_entries, group=self.proc_group)
        locations = {}
        for entries in gathered:
            for dataset_id, sequence, position, node_id in entries:
                locations.setdefault(CacheKey(dataset_id, sequence, position), set()).add(node_id)
        self._remote_locations = locations
        if self.is_node_leader:
            payload = [
                [key.dataset_id, key.sequence, key.position, sorted(node_ids)]
                for key, node_ids in locations.items()
            ]
            temporary = self._remote_registry_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(temporary, self._remote_registry_path)
        dist.barrier(group=self.proc_group)
        self._load_remote_registry(force=True)
        metrics = torch.tensor(
            [
                self.cache_hits_local.value,
                self.cache_hits_remote.value,
                self.cache_misses.value,
                self.total_fetches.value,
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(metrics, group=self.proc_group)
        if self.global_rank == 0:
            LOGGER.info(
                "Synchronized %s cache entries; totals: local=%s remote=%s misses=%s fetches=%s",
                len(locations),
                *metrics.tolist(),
            )

    def _load_remote_registry(self, force=False):
        try:
            mtime = self._remote_registry_path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if not force and mtime == self._remote_registry_mtime:
            return
        payload = json.loads(self._remote_registry_path.read_text())
        self._remote_locations = {
            CacheKey(dataset_id, int(sequence), int(position)): set(node_ids)
            for dataset_id, sequence, position, node_ids in payload
        }
        self._remote_registry_mtime = mtime

    def state_dict(self):
        return {
            "schema_version": 1,
            "dataset_ids": sorted(self.namespaces),
            "remote_backend": self.remote_backend,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("schema_version") != 1:
            raise CacheConfigurationError("Unsupported dataset cache checkpoint schema")
        checkpoint_ids = set(state_dict.get("dataset_ids", ()))
        current_ids = set(self.namespaces)
        if checkpoint_ids and checkpoint_ids != current_ids:
            LOGGER.warning("Discarding stale cache registry state because dataset fingerprints changed")

    def teardown(self, stage=None):
        if getattr(self, "_torn_down", False):
            return
        self._torn_down = True
        for client in self._tcp_clients.values():
            client.close()
        self._tcp_clients.clear()
        executor = getattr(self, "_remote_executor", None)
        if executor is not None:
            executor.shutdown()
        if getattr(self, "is_node_leader", False):
            server = getattr(self, "_server", None)
            if isinstance(server, ProcessTCPCacheServer):
                server.close()
            elif server is not None:
                server.shutdown()
                server.server_close()
                server_thread = getattr(self, "_server_thread", None)
                if server_thread is not None:
                    server_thread.join(timeout=5)
        self.ds.teardown(stage)

    close = teardown


class TCPDatasetCache(DatasetCache):
    def __init__(self, *args, **kwargs):
        kwargs["remote_backend"] = "tcp"
        super().__init__(*args, **kwargs)


class ZarrDatasetCache(DatasetCache):
    def __init__(self, *args, **kwargs):
        kwargs["remote_backend"] = "zarr"
        super().__init__(*args, **kwargs)
