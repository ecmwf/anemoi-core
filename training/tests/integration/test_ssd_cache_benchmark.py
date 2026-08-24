# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark model training with uncached, cold, and warm dataset reads."""

import logging
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from statistics import median
from unittest import mock

import pytorch_lightning as pl
import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.profilers import SimpleProfiler

from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils import dataset_cache as dataset_cache_module
from anemoi.training.utils.dataset_cache import DatasetCache

LOGGER = logging.getLogger(__name__)

_DATASET_CACHE_SETUP = DatasetCache.setup
_DATASET_CACHE_TEARDOWN = DatasetCache.teardown
_DATASET_CACHE_FETCH_MANY = DatasetCache.fetch_many
_HOSTNAME_SUFFIX = "-ab-gpil-ib"
_REMOTE_CHUNK_BYTES = 48 << 20
_REMOTE_PIPELINE_DEPTH = 3
_REMOTE_SERVER_CPUS = 4


def _transfer_snapshot(cache):
    return tuple(
        counter.value
        for counter in (
            cache._benchmark_server_read_ns,
            cache._benchmark_server_encode_ns,
            cache._benchmark_server_send_ns,
            cache._benchmark_client_roundtrip_ns,
            cache._benchmark_client_decode_ns,
            cache._benchmark_wire_bytes,
            cache._benchmark_remote_requests,
        )
    )


def _add(counter, value):
    with counter.get_lock():
        counter.value += value


class _EpochTimer(pl.Callback):
    def __init__(self, remote_ready_paths=None):
        self.durations = []
        self.cache_stats = []
        self.cache_io_seconds = []
        self.transfer_stats = []
        self.dataloader_wait_seconds = 0.0
        self.remote_ready_paths = remote_ready_paths
        self._started_at = None
        self._cache_before = None
        self._cache_io_before = None
        self._transfer_before = None

    @staticmethod
    def _sync():
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _cache_counts(trainer):
        cache = trainer.datamodule
        if not isinstance(cache, DatasetCache):
            return None
        return tuple(
            counter.value
            for counter in (cache.cache_hits_local, cache.cache_hits_remote, cache.cache_misses)
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync()
        self._cache_before = self._cache_counts(trainer)
        cache = trainer.datamodule
        self._cache_io_before = cache.profile_snapshot() if isinstance(cache, DatasetCache) else None
        self._transfer_before = _transfer_snapshot(cache) if hasattr(cache, "_benchmark_wire_bytes") else None
        self._started_at = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        self._sync()
        self.durations.append(time.perf_counter() - self._started_at)
        counts = self._cache_counts(trainer)
        if counts is not None:
            self.cache_stats.append(tuple(after - before for after, before in zip(counts, self._cache_before)))
            self.cache_io_seconds.append(
                tuple(
                    (after - before) / 1e9
                    for after, before in zip(trainer.datamodule.profile_snapshot(), self._cache_io_before)
                )
            )
            if self._transfer_before is not None:
                self.transfer_stats.append(
                    tuple(
                        after - before
                        for after, before in zip(_transfer_snapshot(trainer.datamodule), self._transfer_before)
                    )
                )
        if self.remote_ready_paths is not None and trainer.current_epoch == 0:
            cache = trainer.datamodule
            self.remote_ready_paths[cache.node_id].touch()


class _BenchmarkTrainer(AnemoiTrainer):
    def __init__(self, config, timer, profiler):
        self._timer = timer
        self._profiler = profiler
        super().__init__(config)

    @cached_property
    def profiler(self):
        return self._profiler

    @cached_property
    def callbacks(self):
        return [*super().callbacks, self._timer]


def _run_training(config, cache_root=None, remote_ready_paths=None):
    config = deepcopy(config)
    config.training.max_epochs = 3
    config.dataloader.limit_batches.validation = 0
    config.diagnostics.print_memory_summary = False
    config.diagnostics.log.mlflow.enabled = False
    config.system.hardware.cache_dir = None if cache_root is None else str(cache_root)

    timer = _EpochTimer(remote_ready_paths)
    profiler = SimpleProfiler()
    trainer = _BenchmarkTrainer(config, timer, profiler)
    trainer.datamodule.ds_train.shuffle = False
    trainer.train()
    timer.dataloader_wait_seconds = sum(
        profiler.recorded_durations.get("[_TrainingEpochLoop].train_dataloader_next", ())
    )
    return timer


def _setup_process_cache(self, stage=None):
    rank = dist.get_rank(self.initial_proc_group)
    host = dataset_cache_module.socket.gethostname()
    self.hostname_suffix = _HOSTNAME_SUFFIX
    with mock.patch.object(dataset_cache_module.socket, "gethostname", return_value=f"local-rank-{rank}"):
        result = _DATASET_CACHE_SETUP(self, stage)
    self.hostnames = [host + self.hostname_suffix] * self.world_size
    self._server.close()
    context = multiprocessing.get_context("spawn")
    for name in (
        "server_read_ns",
        "server_encode_ns",
        "server_send_ns",
        "client_roundtrip_ns",
        "client_decode_ns",
        "wire_bytes",
        "remote_requests",
    ):
        setattr(self, f"_benchmark_{name}", context.Value("q", 0))
    counters = tuple(
        getattr(self, f"_benchmark_{name}")
        for name in ("server_read_ns", "server_encode_ns", "server_send_ns")
    )
    entries = {dataset_id: str(namespace.entries_path) for dataset_id, namespace in self.namespaces.items()}
    self._benchmark_server_stop = context.Event()
    ready_parent, ready_child = context.Pipe(duplex=False)
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    reserved_count = min(len(available_cpus) - 1, _REMOTE_SERVER_CPUS * self.world_size)
    reserved_cpus = available_cpus[:reserved_count]
    affinity = reserved_cpus[rank::self.world_size]
    training_affinity = available_cpus[reserved_count:]
    self._benchmark_original_affinity = available_cpus
    self._benchmark_server_process = context.Process(
        target=_serve_shards,
        args=(self.port, entries, self._benchmark_server_stop, ready_child, counters, affinity),
        daemon=True,
    )
    self._benchmark_server_process.start()
    ready_child.close()
    error = ready_parent.recv()
    ready_parent.close()
    if error is not None:
        raise RuntimeError(error)
    if training_affinity:
        os.sched_setaffinity(0, training_affinity)
    LOGGER.info(
        "Rank %s synthetic cache node %s using TCP endpoint %s:%s in process %s on CPUs %s",
        rank,
        self.node_id,
        self._endpoint(self.node_id).host,
        self.port,
        self._benchmark_server_process.pid,
        affinity,
    )
    return result


def _teardown_process_cache(self, stage=None):
    process = getattr(self, "_benchmark_server_process", None)
    if process is not None:
        self._benchmark_server_stop.set()
        try:
            with dataset_cache_module.socket.create_connection(
                (self._endpoint(self.node_id).host, self.port), timeout=1
            ):
                pass
        except OSError:
            pass
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    affinity = getattr(self, "_benchmark_original_affinity", None)
    if affinity:
        os.sched_setaffinity(0, affinity)
    _DATASET_CACHE_TEARDOWN(self, stage)


def _remote_marker(self, node_id, key):
    remote_host = self._raw_hostnames[self._node_leader_rank[node_id]]
    cache_dir = f"cache-{dataset_cache_module.hashlib.sha256(remote_host.encode()).hexdigest()[:12]}"
    namespace = self.namespaces[key.dataset_id]
    return self.cache_root / cache_dir / namespace.token / "committed" / str(key.sequence) / str(key.position)


def _recv_into(sock, size):
    payload = bytearray(size)
    view = memoryview(payload)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if not received:
            return None
        offset += received
    return payload


def _handle_shard_requests(connection, entries, counters):
    server_read_ns, server_encode_ns, server_send_ns = counters
    try:
        while True:
            header = dataset_cache_module._recv_exact(connection, 4)
            if header is None:
                break
            request_payload = dataset_cache_module._recv_exact(
                connection, dataset_cache_module.struct.unpack("!I", header)[0]
            )
            if request_payload is None:
                break
            request = dataset_cache_module.json.loads(request_payload)
            grid = request["grid"]
            grid_indices = slice(grid[0], grid[1], grid[2])
            started = time.perf_counter_ns()
            paths = [
                Path(entries[request["dataset_id"]]) / str(request["sequence"]) / f"{position}.npy"
                for position in request["positions"]
            ]
            if not all(path.exists() for path in paths):
                connection.sendall(b"\x00" + dataset_cache_module.struct.pack("!I", 0) + dataset_cache_module.struct.pack("!Q", 0))
                continue
            first = dataset_cache_module.np.load(paths[0], allow_pickle=False, mmap_mode="r")[..., grid_indices]
            value = dataset_cache_module.np.empty((len(paths), *first.shape), dtype=first.dtype)
            value[0] = first
            for index, path in enumerate(paths[1:], start=1):
                value[index] = dataset_cache_module.np.load(path, allow_pickle=False, mmap_mode="r")[..., grid_indices]
            _add(server_read_ns, time.perf_counter_ns() - started)
            started = time.perf_counter_ns()
            metadata = dataset_cache_module.json.dumps(
                {"dtype": value.dtype.str, "shape": value.shape}, separators=(",", ":")
            ).encode()
            _add(server_encode_ns, time.perf_counter_ns() - started)
            started = time.perf_counter_ns()
            connection.sendall(
                b"\x01"
                + dataset_cache_module.struct.pack("!I", len(metadata))
                + dataset_cache_module.struct.pack("!Q", value.nbytes)
                + metadata
            )
            connection.sendall(memoryview(value).cast("B"))
            _add(server_send_ns, time.perf_counter_ns() - started)
    except (ConnectionResetError, BrokenPipeError, OSError, ValueError, KeyError):
        pass
    finally:
        connection.close()


def _serve_shards(port, entries, stop, ready, counters, affinity):
    try:
        if affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, affinity)
        server = dataset_cache_module.socket.socket(dataset_cache_module.socket.AF_INET, dataset_cache_module.socket.SOCK_STREAM)
        server.setsockopt(dataset_cache_module.socket.SOL_SOCKET, dataset_cache_module.socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(64)
        server.settimeout(0.5)
        ready.send(None)
    except Exception as error:
        ready.send(f"remote cache server failed to start: {error}")
        ready.close()
        return
    ready.close()
    try:
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            thread = dataset_cache_module.threading.Thread(
                target=_handle_shard_requests,
                args=(connection, entries, counters),
                daemon=True,
            )
            thread.start()
    finally:
        server.close()


def _fetch_remote_shard(self, node_id, key, positions, grid_indices, slot):
    client_key = (os.getpid(), node_id, slot)
    client = self._tcp_clients.get(client_key)
    if client is None:
        client = self._tcp_clients[client_key] = dataset_cache_module.TCPCacheClient(self._endpoint(node_id))
    request = dataset_cache_module.json.dumps(
        {
            "dataset_id": key.dataset_id,
            "sequence": key.sequence,
            "positions": positions,
            "grid": [grid_indices.start, grid_indices.stop, grid_indices.step],
        }
    ).encode()
    for attempt in range(2):
        try:
            if client._sock is None:
                client._connect()
            started = time.perf_counter_ns()
            client._sock.sendall(dataset_cache_module.struct.pack("!I", len(request)) + request)
            header = dataset_cache_module._recv_exact(client._sock, 13)
            if header is None:
                raise ConnectionError("server closed connection")
            if header[0] == 0:
                raise dataset_cache_module.RemoteCacheMiss(key)
            metadata = dataset_cache_module._recv_exact(
                client._sock, dataset_cache_module.struct.unpack("!I", header[1:5])[0]
            )
            payload = _recv_into(client._sock, dataset_cache_module.struct.unpack("!Q", header[5:])[0])
            if payload is None:
                raise ConnectionError("server closed connection")
            _add(self._benchmark_client_roundtrip_ns, time.perf_counter_ns() - started)
            _add(self._benchmark_wire_bytes, len(payload))
            _add(self._benchmark_remote_requests, 1)
            started = time.perf_counter_ns()
            metadata = dataset_cache_module.json.loads(metadata)
            value = dataset_cache_module.np.frombuffer(payload, dtype=metadata["dtype"]).reshape(metadata["shape"])
            _add(self._benchmark_client_decode_ns, time.perf_counter_ns() - started)
            return value
        except dataset_cache_module.RemoteCacheMiss:
            raise
        except (ConnectionResetError, BrokenPipeError, ConnectionError, OSError) as error:
            client.close()
            if attempt:
                raise dataset_cache_module.RemoteCacheUnavailable(str(error)) from error
    raise dataset_cache_module.RemoteCacheUnavailable(str(client.endpoint))


def _fetch_chunk_lane(self, remote_node, key, positions, chunks, slot):
    return [
        (index, _fetch_remote_shard(self, remote_node, key, positions, chunk, slot))
        for index, chunk in chunks
    ]


def _fetch_many_remote(self, dataset_id, sequence, positions, grid_indices=None):
    if not all(path.exists() for path in self._benchmark_remote_ready_paths):
        return _DATASET_CACHE_FETCH_MANY(self, dataset_id, sequence, positions, grid_indices)
    dataset_id = self._resolve_dataset_id(dataset_id)
    namespace = self.namespaces[dataset_id]
    normalized = self._normalize_positions(namespace, sequence, positions)
    sequence = int(sequence)
    key = dataset_cache_module.CacheKey(dataset_id, sequence, normalized[0])
    remote_node = (self.node_id + 1) % self.num_nodes
    if normalized and isinstance(grid_indices, slice) and all(
        _remote_marker(self, remote_node, dataset_cache_module.CacheKey(dataset_id, sequence, position)).exists()
        for position in normalized
    ):
        started = time.perf_counter_ns() if self.profile_io else 0
        try:
            reference = namespace.fetch_local(sequence, normalized[0])
            start, stop, step = grid_indices.indices(reference.shape[-1])
            if step != 1:
                raise ValueError("remote benchmark requires a contiguous grid shard")
            bytes_per_gridpoint = len(normalized) * reference[..., :1].nbytes
            chunk_size = max(1, _REMOTE_CHUNK_BYTES // bytes_per_gridpoint)
            chunks = [slice(offset, min(offset + chunk_size, stop)) for offset in range(start, stop, chunk_size)]
            executor = getattr(self, "_benchmark_remote_executor", None)
            if executor is None:
                executor = self._benchmark_remote_executor = ThreadPoolExecutor(max_workers=_REMOTE_PIPELINE_DEPTH)
            indexed_chunks = list(enumerate(chunks))
            futures = [
                executor.submit(
                    _fetch_chunk_lane,
                    self,
                    remote_node,
                    key,
                    normalized,
                    indexed_chunks[slot::_REMOTE_PIPELINE_DEPTH],
                    slot,
                )
                for slot in range(min(_REMOTE_PIPELINE_DEPTH, len(chunks)))
            ]
            values = [item for future in futures for item in future.result()]
            values.sort(key=lambda item: item[0])
            arrays = [array for _, array in values]
            value = dataset_cache_module.np.empty((*arrays[0].shape[:-1], stop - start), dtype=arrays[0].dtype)
            offset = 0
            for chunk in arrays:
                value[..., offset : offset + chunk.shape[-1]] = chunk
                offset += chunk.shape[-1]
        except (dataset_cache_module.RemoteCacheMiss, dataset_cache_module.RemoteCacheUnavailable):
            pass
        else:
            if started:
                self.profile_remote_ns.value += time.perf_counter_ns() - started
            self.total_fetches.value += len(normalized)
            self.cache_hits_remote.value += len(normalized)
            if started:
                self.profile_fetch_many_ns.value += time.perf_counter_ns() - started
            return value
        if started:
            self.profile_remote_ns.value += time.perf_counter_ns() - started
    return _DATASET_CACHE_FETCH_MANY(self, dataset_id, sequence, positions, grid_indices)


def _run_remote_training(config, cache_root, monkeypatch):
    remote_ready_paths = tuple(
        cache_root / f".remote-ready-{node_id}"
        for node_id in range(int(config.system.hardware.num_gpus_per_node))
    )
    for path in remote_ready_paths:
        path.unlink(missing_ok=True)
    with monkeypatch.context() as patch:
        patch.setattr(DatasetCache, "setup", _setup_process_cache)
        patch.setattr(DatasetCache, "teardown", _teardown_process_cache)
        patch.setattr(DatasetCache, "close", _teardown_process_cache)
        patch.setattr(DatasetCache, "fetch_many", _fetch_many_remote)
        patch.setattr(DatasetCache, "_benchmark_remote_ready_paths", remote_ready_paths, raising=False)
        return _run_training(config, cache_root, remote_ready_paths)


def test_remote_shard_server(tmp_path):
    entries = tmp_path / "entries" / "0"
    entries.mkdir(parents=True)
    arrays = [dataset_cache_module.np.arange(32, dtype="float32").reshape(2, 16) + offset for offset in (0, 100)]
    for position, array in enumerate(arrays):
        dataset_cache_module.np.save(entries / f"{position}.npy", array)

    context = multiprocessing.get_context("spawn")
    counters = tuple(context.Value("q", 0) for _ in range(3))
    stop = context.Event()
    ready_parent, ready_child = context.Pipe(duplex=False)
    probe = dataset_cache_module.socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    process = context.Process(
        target=_serve_shards,
        args=(port, {"dataset": str(entries.parent)}, stop, ready_child, counters, []),
        daemon=True,
    )
    process.start()
    ready_child.close()
    assert ready_parent.recv() is None
    ready_parent.close()
    try:
        request = dataset_cache_module.json.dumps(
            {"dataset_id": "dataset", "sequence": 0, "positions": [0, 1], "grid": [4, 12, 1]}
        ).encode()
        with dataset_cache_module.socket.create_connection(("127.0.0.1", port)) as client:
            client.sendall(dataset_cache_module.struct.pack("!I", len(request)) + request)
            header = dataset_cache_module._recv_exact(client, 13)
            assert header[0] == 1
            metadata = dataset_cache_module._recv_exact(
                client, dataset_cache_module.struct.unpack("!I", header[1:5])[0]
            )
            payload = _recv_into(client, dataset_cache_module.struct.unpack("!Q", header[5:])[0])
        metadata = dataset_cache_module.json.loads(metadata)
        result = dataset_cache_module.np.frombuffer(payload, dtype=metadata["dtype"]).reshape(metadata["shape"])
        dataset_cache_module.np.testing.assert_array_equal(
            result, dataset_cache_module.np.stack([array[..., 4:12] for array in arrays])
        )
        assert all(counter.value > 0 for counter in counters)
    finally:
        stop.set()
        with dataset_cache_module.socket.create_connection(("127.0.0.1", port)):
            pass
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


@pytest.mark.multigpu
@pytest.mark.slow
def test_ssd_cache_training_runtime(
    benchmark_config: tuple[DictConfig, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare uncached, local SSD cache, and local TCP cache training."""
    config, test_case = benchmark_config
    required_gpus = int(config.system.hardware.num_gpus_per_node)
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(f"benchmark requires {required_gpus} GPUs, found {torch.cuda.device_count()}")
    cache_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    run_id = os.environ.get("SLURM_JOB_ID", "local")
    local_cache_root = cache_root / f"anemoi-cache-benchmark-local-{run_id}"
    remote_cache_root = cache_root / f"anemoi-cache-benchmark-remote-{run_id}"
    local_cache_root.mkdir(parents=True, exist_ok=True)
    remote_cache_root.mkdir(parents=True, exist_ok=True)

    uncached = _run_training(config)
    previous_profile_setting = os.environ.get("ANEMOI_DATASET_CACHE_PROFILE")
    os.environ["ANEMOI_DATASET_CACHE_PROFILE"] = "1"
    try:
        cached = _run_training(config, local_cache_root)
        remote = _run_remote_training(config, remote_cache_root, monkeypatch)
    finally:
        if previous_profile_setting is None:
            os.environ.pop("ANEMOI_DATASET_CACHE_PROFILE")
        else:
            os.environ["ANEMOI_DATASET_CACHE_PROFILE"] = previous_profile_setting

    assert len(uncached.durations) == len(cached.durations) == len(remote.durations) == 3
    assert cached.cache_stats[0][2] > 0, "cold epoch did not populate the cache"
    assert all(local + remote > 0 and misses == 0 for local, remote, misses in cached.cache_stats[1:]), (
        f"cache was not warm after epoch 0: {cached.cache_stats}"
    )
    assert remote.cache_stats[0][2] > 0, "cold remote epoch did not populate the process caches"
    assert all(remote_hits > local_hits for local_hits, remote_hits, _ in remote.cache_stats[1:]), (
        f"TCP cache was not used after epoch 0: {remote.cache_stats}"
    )

    no_cache_seconds = median(uncached.durations)
    cold_cache_seconds = cached.durations[0]
    warm_cache_seconds = median(cached.durations[1:])
    cold_remote_seconds = remote.durations[0]
    warm_remote_seconds = median(remote.durations[1:])
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("LOCAL_RANK", 0))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        LOGGER.info(
            "%s SSD cache benchmark: no_cache=%.3fs cold=%.3fs warm=%.3fs "
            "cold_speedup=%.3fx warm_speedup=%.3fx cache_stats=%s "
            "cache_io_seconds(local,remote,source,store,fetch_many)=%s "
            "dataloader_wait(no_cache,cached)=%s TCP: cold=%.3fs warm=%.3fs "
            "cold_speedup=%.3fx warm_speedup=%.3fx cache_stats=%s "
            "cache_io_seconds(local,remote,source,store,fetch_many)=%s dataloader_wait=%.3fs "
            "transfer(server_read_ns,encode_ns,send_ns,roundtrip_ns,decode_ns,bytes,requests)=%s",
            test_case,
            no_cache_seconds,
            cold_cache_seconds,
            warm_cache_seconds,
            no_cache_seconds / cold_cache_seconds,
            no_cache_seconds / warm_cache_seconds,
            cached.cache_stats,
            cached.cache_io_seconds,
            (uncached.dataloader_wait_seconds, cached.dataloader_wait_seconds),
            cold_remote_seconds,
            warm_remote_seconds,
            no_cache_seconds / cold_remote_seconds,
            no_cache_seconds / warm_remote_seconds,
            remote.cache_stats,
            remote.cache_io_seconds,
            remote.dataloader_wait_seconds,
            remote.transfer_stats,
        )
        shutil.rmtree(local_cache_root)
        shutil.rmtree(remote_cache_root)
    if dist.is_initialized():
        dist.barrier()