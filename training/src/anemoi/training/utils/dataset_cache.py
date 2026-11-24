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
import io
import json
import logging
import os
import pickle
import shutil
import socket
import threading
from collections.abc import Callable
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import Value
from pathlib import Path
from typing import BinaryIO

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


def decode_cache_array(data) -> np.ndarray:
    """Decode an uncompressed .npy header followed by the blosc-compressed array data."""
    view = memoryview(data)
    # a version 1.0 .npy header: 6-byte magic string, 2-byte version, 2-byte header length
    header_end = 10 + int.from_bytes(view[8:10], "little")
    source = io.BytesIO(view[:header_end])
    np.lib.format.read_magic(source)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
    value = np.empty(shape, dtype=dtype, order="F" if fortran_order else "C")
    _BLOSC.decode(view[header_end:], out=value)
    return value


def load_cache_array(path):
    """Read an uncompressed .npy header followed by the blosc-compressed array data."""
    return decode_cache_array(path.read_bytes())


def encode_cache_array(value: np.ndarray) -> bytes:
    """Return an uncompressed .npy header followed by the blosc-compressed array data."""
    value = np.ascontiguousarray(value)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(value))
    return header.getvalue() + _BLOSC.encode(value)


def encode_cache_header(shape: tuple[int, ...], dtype: np.dtype | str) -> bytes:
    """Return the uncompressed .npy header of a C-order record with this shape and dtype."""
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": tuple(shape)},
    )
    return header.getvalue()


def save_cache_array(target, value):
    target.write(encode_cache_array(value))


def cache_entry_path(entries_path, sequence, position, grid_id="all") -> Path:
    return Path(entries_path) / grid_id / str(sequence) / f"{position}.npy.blosc"


def read_cache_entry(entries_path, sequence, position, grid_id="all"):
    entry = cache_entry_path(entries_path, sequence, position, grid_id)
    return load_cache_array(entry) if entry.exists() else None


def read_cache_payload(entries_path, sequence, position, grid_id="all") -> bytes | None:
    """Return the compressed bytes of a record, or None if it is not in the cache."""
    try:
        return cache_entry_path(entries_path, sequence, position, grid_id).read_bytes()
    except FileNotFoundError:
        return None


class SourceChunks:
    """Read the records of a dataset as the compressed chunks of its zarr array.

    In the source zarr of an analysis dataset, each chunk is often one date with all the
    variables, ensemble members and grid points, compressed with blosc. Then one chunk holds the
    same data as one cache record. On a cache miss, the cache writes the chunk bytes as the record.
    It does not compress the record again. blosc data holds its own settings, thus
    decode_cache_array can decode these records, also on other nodes.

    from_dataset returns None for all other datasets. Then the cache reads each record with the
    dataset reader and compresses it.
    """

    def __init__(self, array: object, indices: list[int] | None, missing: set[int]) -> None:
        self.array = array
        self.indices = indices  # position in the dataset -> index in the zarr array; None: the same
        self.missing = missing
        self.header = encode_cache_header(array.shape[1:], array.dtype)

    @classmethod
    def from_dataset(cls, data: object) -> "SourceChunks | None":
        """Return a SourceChunks for a zarr dataset or a date subset of one, else None."""
        indices = None
        # open_dataset with start, end or frequency gives a Subset: a list of dates of the zarr
        while type(data).__name__ == "Subset" and hasattr(data, "indices") and hasattr(data, "dataset"):
            subset = [int(index) for index in data.indices]
            indices = subset if indices is None else [subset[index] for index in indices]
            data = data.dataset
        if type(data).__name__ != "GriddedZarr":
            return None
        array = getattr(data, "data", None)
        if (
            array is None
            or getattr(getattr(array, "compressor", None), "codec_id", None) != "blosc"
            or getattr(array, "filters", None)
            or getattr(array, "order", None) != "C"
            or len(array.shape) != 4
            or tuple(array.chunks) != (1, *array.shape[1:])
            or not hasattr(array, "_chunk_key")
            or not hasattr(array, "chunk_store")
        ):
            return None
        return cls(array, indices, set(getattr(data, "missing", ())))

    def covers_grid(self, grid_indices: slice | np.ndarray | None) -> bool:
        """Return True if grid_indices select all the grid points, in their order."""
        if grid_indices is None:
            return True
        if isinstance(grid_indices, slice):
            size = self.array.shape[-1]
            return grid_indices.indices(size) == (0, size, 1)
        return False

    def read(self, position: int) -> bytes | None:
        """Return the record of a position (a .npy header and the chunk bytes), or None if there is no chunk."""
        index = position if self.indices is None else self.indices[position]
        if index in self.missing:
            return None
        try:
            chunk = self.array.chunk_store[self.array._chunk_key((index, 0, 0, 0))]
        except KeyError:  # the chunk is not in the store: zarr gives the fill value
            return None
        return self.header + chunk


def _is_capacity_error(error: OSError) -> bool:
    errnos = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
    messages = ("disk quota exceeded", "no space left on device", "not enough free space")
    return error.errno in errnos or any(message in str(error).lower() for message in messages)


class CacheFullError(OSError):
    """The node cache has reached its size limit, or its disk is full."""


class NodeCacheBudget:
    """Size limit for the cache of one node, shared by all processes on the node.

    A small file holds the number of bytes that the cache uses. Each change to this file
    occurs under an flock. Thus, all ranks and dataloader workers on the node use the same
    budget. A writer reserves space before it writes a record. Thus, concurrent writers
    cannot go over the limit.

    The budget also keeps ``min_free_bytes`` free on the disk. This limit includes files
    that are not in the cache, for example checkpoints and logs.
    """

    def __init__(self, path: Path, limit_bytes: int, min_free_bytes: int = 0):
        self.path = Path(path)
        self.limit_bytes = int(limit_bytes)
        self.min_free_bytes = int(min_free_bytes)

    def reset(self) -> None:
        self.path.write_bytes((0).to_bytes(8, "little"))

    @contextmanager
    def _locked(self) -> Iterator[BinaryIO]:
        with self.path.open("r+b") as counter:
            fcntl.flock(counter, fcntl.LOCK_EX)
            try:
                yield counter
            finally:
                fcntl.flock(counter, fcntl.LOCK_UN)

    def used(self) -> int:
        with self._locked() as counter:
            return int.from_bytes(counter.read(8), "little")

    def _add(self, nbytes: int, limit: bool) -> bool:
        with self._locked() as counter:
            used = int.from_bytes(counter.read(8), "little")
            if limit and used + nbytes > self.limit_bytes:
                return False
            if limit and shutil.disk_usage(self.path.parent).free - nbytes < self.min_free_bytes:
                return False
            counter.seek(0)
            counter.write(max(used + nbytes, 0).to_bytes(8, "little"))
            return True

    def reserve(self, nbytes: int) -> bool:
        """Reserve ``nbytes``. If this goes over a limit, reserve nothing and return False."""
        return self._add(nbytes, limit=True)

    def release(self, nbytes: int) -> None:
        self._add(-nbytes, limit=False)


class DatasetCacheNamespace:
    """Atomic file-per-record cache for one dataset reader."""

    def __init__(self, root: Path, dataset_id: str, reader, budget: NodeCacheBudget | None = None):
        self.dataset_id = dataset_id
        self.reader = reader
        self.budget = budget
        self.num_sequences = int(reader.num_sequences)
        self.sequence_lengths = [int(reader.sequence_length(index)) for index in range(self.num_sequences)]
        self.path = root / hashlib.sha256(dataset_id.encode()).hexdigest()[:20]
        self.entries_path = self.path / "entries"
        # only analysis datasets: one sequence, and the position is the date
        self.source_chunks = (
            SourceChunks.from_dataset(getattr(reader, "data", None)) if self.num_sequences == 1 else None
        )
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
        return self.store_payload(sequence, position, lambda: encode_cache_array(value), grid_id, sync=sync)

    def store_payload(
        self,
        sequence: int,
        position: int,
        payload: bytes | Callable[[], bytes],
        grid_id: str = "all",
        sync: bool = False,
    ) -> bool:
        """Write one record. payload is the record bytes, or a function that returns them.

        The function is called only if the record is not in the cache.
        """
        entry, lock = self.paths(sequence, position, grid_id)
        with self.lock(lock):
            if entry.exists():
                return False
            payload = payload() if callable(payload) else payload
            if self.budget is not None and not self.budget.reserve(len(payload)):
                msg = (
                    f"cache limit of {self.budget.limit_bytes / 1e9:.1f} GB or free-space limit of "
                    f"{self.budget.min_free_bytes / 1e9:.1f} GB reached, cannot add {len(payload) / 1e9:.2f} GB"
                )
                raise CacheFullError(msg)
            entry.parent.mkdir(parents=True, exist_ok=True)
            temporary = entry.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                with temporary.open("wb") as target:
                    target.write(payload)
                    if sync:
                        target.flush()
                        os.fsync(target.fileno())
                os.replace(temporary, entry)
                return True
            except OSError as error:
                temporary.unlink(missing_ok=True)
                if self.budget is not None:
                    self.budget.release(len(payload))
                if _is_capacity_error(error):
                    raise CacheFullError(str(error)) from error
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

    def __init__(
        self,
        ds,
        cache_root,
        hostname_suffix=None,
        proc_group=None,
        async_writes=False,
        max_fraction: float = 0.9,
    ):
        super().__init__()
        self.ds = ds  # must be set before any attribute access can reach __getattr__
        self.cache_root = Path(cache_root)
        self.hostname_suffix = hostname_suffix or ""
        self.initial_proc_group = proc_group
        self.async_writes = async_writes
        if not 0 < max_fraction <= 1:
            msg = f"max_fraction must be in (0, 1], got {max_fraction}"
            raise ValueError(msg)
        self.max_fraction = max_fraction
        self.budget = None  # set in setup(); None means no size limit
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
        self.cache_source_records = Value("q", 0)
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
            # Use only fields that are the same in every process. Do not use the dataset metadata:
            # for a cloned dataset, the dates can be invalid, and then the metadata changes from
            # one process to the next. Each rank then gets a different dataset_id and cache.
            identity = {
                "name": name,
                "shape": tuple(int(size) for size in reader.data.shape),
                "dtype": str(reader.data.dtype),
                "source": str(getattr(reader.data, "path", type(reader.data).__name__)),
                "variables": list(reader.variables),
                "resolution": str(reader.resolution),
            }
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]
            dataset_id = f"{name}:{digest}"
            self.namespaces[dataset_id] = DatasetCacheNamespace(
                self.cache_path,
                dataset_id,
                reader,
                budget=self.budget,
            )
            reader.set_cache(self, dataset_id)
            if getattr(self, "global_rank", 0) == 0:
                LOGGER.info(
                    "Dataset cache for %s: %s",
                    dataset_id,
                    (
                        "misses store the compressed zarr chunks of the source"
                        if self.namespaces[dataset_id].source_chunks is not None
                        else "misses read the dataset and compress each record"
                    ),
                )
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
        self._check_hostnames()
        self.cache_path = self.cache_root / f"cache-{hashlib.sha256(host.encode()).hexdigest()[:12]}"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        disk_bytes = shutil.disk_usage(self.cache_root).total
        # the cache and all other files together must not use more than max_fraction of the disk
        limit_bytes = int(self.max_fraction * disk_bytes)
        self.budget = NodeCacheBudget(self.cache_path / "usage", limit_bytes, min_free_bytes=disk_bytes - limit_bytes)
        if self.is_node_leader:
            shutil.rmtree(self.cache_path, ignore_errors=True)
            self.cache_path.mkdir(parents=True)
            self.budget.reset()
            LOGGER.info(
                "Dataset cache on node %d is limited to %.1f GB (%.0f%% of %.1f GB)",
                self.node_id,
                self.budget.limit_bytes / 1e9,
                100 * self.max_fraction,
                disk_bytes / 1e9,
            )
        dist.barrier(group=self.proc_group)
        self.initialized = True
        self._dataset(self.ds.ds_train)
        self._check_dataset_ids()
        if self.is_node_leader:
            entries = {name: str(namespace.entries_path) for name, namespace in self.namespaces.items()}
            self.server = CacheServer(entries)
            self.server.start()
        ports = [None] * self.world_size
        dist.all_gather_object(ports, self.server.port if self.is_node_leader else None, group=self.proc_group)
        self._ports = {node: ports[rank] for node, rank in self._leaders.items()}

    def _check_hostnames(self) -> None:
        """Make sure that each rank can find the address of the cache server on each other node.

        ZeroMQ does not show an error when it cannot find the address of a host. It tries to
        connect again and again, and each remote read waits until the timeout. Thus, an incorrect
        hostname_suffix makes the training slow, but shows only a timeout. This check stops
        the run at the start and shows the hostname that is not correct.
        """
        unresolved = []
        for node, rank in self._leaders.items():
            if node == self.node_id:
                continue
            try:
                socket.getaddrinfo(self.hostnames[rank], None)
            except socket.gaierror as error:
                unresolved.append(f"{self.hostnames[rank]} ({error})")
        # all ranks must stop together, else the other ranks wait in a collective
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, unresolved, group=self.proc_group)
        failures = sorted({name for names in gathered for name in names})
        if failures:
            msg = (
                f"Cannot find the address of the dataset cache server on other nodes: {', '.join(failures)}. "
                f"Make sure that system.hardware.hostname_suffix ({self.hostname_suffix!r}) is correct for "
                "this system, or set it to null."
            )
            raise RuntimeError(msg)

    def _check_dataset_ids(self):
        """Make sure that all ranks use the same dataset_ids, else they cannot share records."""
        dataset_ids = [None] * self.world_size
        dist.all_gather_object(dataset_ids, sorted(self.namespaces), group=self.proc_group)
        if any(ids != dataset_ids[0] for ids in dataset_ids):
            msg = f"Ranks do not agree on the cached dataset_ids: {dataset_ids}"
            raise RuntimeError(msg)

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

    def _store_records(self, dataset_id, sequence, positions, values, grid_id, sync=False, encoded=False):
        if self.cache_full.value:
            return
        namespace = self.namespaces[dataset_id]
        try:
            for position, value in zip(positions, values):
                if encoded:
                    namespace.store_payload(sequence, position, value, grid_id, sync=sync)
                else:
                    namespace.store(sequence, position, value, grid_id, sync=sync)
        except CacheFullError as error:
            # the cache stays locked at its size: no records are added or removed after this
            with self.cache_full.get_lock():
                first = not self.cache_full.value
                self.cache_full.value = True
            if first:
                LOGGER.warning("Dataset cache on node %s is full, no more records are cached: %s", self.node_id, error)

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

    def read_source_records(
        self,
        dataset_id: str,
        sequence: int,
        positions: list[int] | slice,
        grid_indices: slice | np.ndarray | None = None,
    ) -> list[np.ndarray] | None:
        """Read records from the compressed chunks of the source zarr, and add them to the cache.

        Return the decoded records. Return None if the source cannot give the records in this way
        (see SourceChunks). Then the caller reads the records with the dataset reader and calls
        store_records.
        """
        namespace = self.namespaces[dataset_id]
        chunks = namespace.source_chunks
        if chunks is None or not chunks.covers_grid(grid_indices):
            return None
        positions = self._positions(namespace, sequence, positions)
        payloads = [chunks.read(namespace.normalize(sequence, position)[1]) for position in positions]
        if any(payload is None for payload in payloads):
            return None
        values = [decode_cache_array(payload) for payload in payloads]
        # the write does not compress, thus it is short: do it now, also when async_writes is set
        self._store_records(dataset_id, sequence, positions, payloads, self._grid_id(grid_indices), encoded=True)
        self._increment(self.cache_source_records, len(positions))
        return values

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
            "Rank %d had %d fetches (%d local hits, %d remote hits, %d misses, %d remote errors, "
            "%d records from source chunks)",
            self.global_rank,
            self.total_fetches.value,
            self.cache_hits_local.value,
            self.cache_hits_remote.value,
            self.cache_misses.value,
            self.cache_remote_errors.value,
            self.cache_source_records.value,
        )

    def teardown(self, stage=None):
        self.close_writer()
        for connection in self._connections.values():
            connection.close()
        if self.is_node_leader:
            self.server.close()
        self._view_path.unlink(missing_ok=True)
        self.ds.teardown(stage)
