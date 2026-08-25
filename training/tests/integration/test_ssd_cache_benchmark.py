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
import os
import shutil
import tempfile
import time
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
_DATASET_CACHE_CHECK = DatasetCache.check_cache
_HOSTNAME_SUFFIX = "-ab-gpil-ib"


class _EpochTimer(pl.Callback):
    def __init__(self, remote_ready_paths=None):
        self.durations = []
        self.cache_stats = []
        self.dataloader_wait_seconds = 0.0
        self.remote_ready_paths = remote_ready_paths
        self._started_at = None
        self._cache_before = None

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
        self._started_at = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        self._sync()
        self.durations.append(time.perf_counter() - self._started_at)
        counts = self._cache_counts(trainer)
        if counts is not None:
            self.cache_stats.append(tuple(after - before for after, before in zip(counts, self._cache_before)))
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
    LOGGER.info(
        "Rank %s synthetic cache node %s using ZeroMQ endpoint %s:%s",
        rank,
        self.node_id,
        self._endpoint(self.node_id).host,
        self.server.port,
    )
    return result


def _remote_marker(self, node_id, key):
    cache_dir = f"cache-{dataset_cache_module.hashlib.sha256(f'local-rank-{node_id}'.encode()).hexdigest()[:12]}"
    namespace = self.namespaces[key.dataset_id]
    return (
        self.cache_root
        / cache_dir
        / namespace.path.name
        / "committed"
        / key.grid_id
        / str(key.sequence)
        / str(key.position)
    )


def _check_cache_remote(self, dataset_id, sequence, positions, grid_indices=None):
    if not all(path.exists() for path in self._benchmark_remote_ready_paths):
        return _DATASET_CACHE_CHECK(self, dataset_id, sequence, positions, grid_indices)
    normalized = self._positions(self.namespaces[dataset_id], sequence, positions)
    sequence = int(sequence)
    grid_id = self._grid_id(grid_indices)
    remote_node = (self.node_id + 1) % len(self._leaders)
    if normalized and isinstance(grid_indices, slice) and all(
        _remote_marker(self, remote_node, dataset_cache_module.CacheKey(dataset_id, sequence, position, grid_id)).exists()
        for position in normalized
    ):
        self._locations.update(
            {
                dataset_cache_module.CacheKey(dataset_id, sequence, position, grid_id): {remote_node}
                for position in normalized
            }
        )
    return _DATASET_CACHE_CHECK(self, dataset_id, sequence, positions, grid_indices)


def _run_remote_training(config, cache_root, monkeypatch):
    remote_ready_paths = tuple(
        cache_root / f".remote-ready-{node_id}"
        for node_id in range(int(config.system.hardware.num_gpus_per_node))
    )
    for path in remote_ready_paths:
        path.unlink(missing_ok=True)
    with monkeypatch.context() as patch:
        patch.setattr(DatasetCache, "setup", _setup_process_cache)
        patch.setattr(DatasetCache, "check_cache", _check_cache_remote)
        patch.setattr(DatasetCache, "_benchmark_remote_ready_paths", remote_ready_paths, raising=False)
        return _run_training(config, cache_root, remote_ready_paths)


@pytest.mark.multigpu
@pytest.mark.slow
def test_ssd_cache_training_remote_runtime(
    benchmark_config: tuple[DictConfig, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Benchmark cold and warm all-remote ZeroMQ cache training."""
    config, test_case = benchmark_config
    required_gpus = int(config.system.hardware.num_gpus_per_node)
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(f"benchmark requires {required_gpus} GPUs, found {torch.cuda.device_count()}")

    cache_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / (
        f"anemoi-cache-benchmark-remote-only-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    remote = _run_remote_training(config, cache_root, monkeypatch)

    assert len(remote.durations) == 3
    assert remote.cache_stats[0][2] > 0, "cold epoch did not populate the process caches"
    assert all(remote_hits > local_hits for local_hits, remote_hits, _ in remote.cache_stats[1:]), (
        f"ZeroMQ cache was not used after epoch 0: {remote.cache_stats}"
    )

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("LOCAL_RANK", 0))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(
            f"{test_case} remote SSD cache: cold={remote.durations[0]:.3f}s "
            f"warm={median(remote.durations[1:]):.3f}s cache_stats={remote.cache_stats} "
            f"dataloader_wait={remote.dataloader_wait_seconds:.3f}s",
            flush=True,
        )
        shutil.rmtree(cache_root)
    if dist.is_initialized():
        dist.barrier()


@pytest.mark.multigpu
@pytest.mark.slow
def test_ssd_cache_training_runtime(
    benchmark_config: tuple[DictConfig, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare uncached, local SSD cache, and local ZeroMQ cache training."""
    config, test_case = benchmark_config
    required_gpus = int(config.system.hardware.num_gpus_per_node)
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(f"benchmark requires {required_gpus} GPUs, found {torch.cuda.device_count()}")
    cache_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    run_id = os.environ.get("SLURM_JOB_ID", "local")
    local_cache_root = cache_root / f"anemoi-cache-benchmark-local-{run_id}"
    local_cache_root.mkdir(parents=True, exist_ok=True)

    uncached = _run_training(config)
    cached = _run_training(config, local_cache_root)

    assert len(uncached.durations) == len(cached.durations) == 3
    assert cached.cache_stats[0][2] > 0, "cold epoch did not populate the cache"

    no_cache_seconds = median(uncached.durations)
    cold_cache_seconds = cached.durations[0]
    warm_cache_seconds = median(cached.durations[1:])
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("LOCAL_RANK", 0))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        LOGGER.info(
            "%s SSD cache benchmark: no_cache=%.3fs cold=%.3fs warm=%.3fs "
            "cold_speedup=%.3fx warm_speedup=%.3fx cache_stats=%s "
            "dataloader_wait(no_cache,cached)=%s",
            test_case,
            no_cache_seconds,
            cold_cache_seconds,
            warm_cache_seconds,
            no_cache_seconds / cold_cache_seconds,
            no_cache_seconds / warm_cache_seconds,
            cached.cache_stats,
            (uncached.dataloader_wait_seconds, cached.dataloader_wait_seconds),
        )
        shutil.rmtree(local_cache_root)
    if dist.is_initialized():
        dist.barrier()