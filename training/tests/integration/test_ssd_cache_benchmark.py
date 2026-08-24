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

import pytorch_lightning as pl
import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.profilers import SimpleProfiler

from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.dataset_cache import DatasetCache

LOGGER = logging.getLogger(__name__)


class _EpochTimer(pl.Callback):
    def __init__(self):
        self.durations = []
        self.cache_stats = []
        self.cache_io_seconds = []
        self.dataloader_wait_seconds = 0.0
        self._started_at = None
        self._cache_before = None
        self._cache_io_before = None

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


def _run_training(config, cache_root=None):
    config = deepcopy(config)
    config.training.max_epochs = 3
    config.dataloader.limit_batches.validation = 0
    config.diagnostics.print_memory_summary = False
    config.diagnostics.log.mlflow.enabled = False
    config.system.hardware.cache_dir = None if cache_root is None else str(cache_root)

    timer = _EpochTimer()
    profiler = SimpleProfiler()
    trainer = _BenchmarkTrainer(config, timer, profiler)
    trainer.datamodule.ds_train.shuffle = False
    trainer.train()
    timer.dataloader_wait_seconds = sum(
        profiler.recorded_durations.get("[_TrainingEpochLoop].train_dataloader_next", ())
    )
    return timer


@pytest.mark.multigpu
@pytest.mark.slow
def test_ssd_cache_training_runtime(benchmark_config: tuple[DictConfig, str]) -> None:
    """Compare full GraphTransformer training epochs with and without SSD caching."""
    config, test_case = benchmark_config
    required_gpus = int(config.system.hardware.num_gpus_per_node)
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(f"benchmark requires {required_gpus} GPUs, found {torch.cuda.device_count()}")
    cache_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / (
        f"anemoi-cache-benchmark-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    uncached = _run_training(config)
    previous_profile_setting = os.environ.get("ANEMOI_DATASET_CACHE_PROFILE")
    os.environ["ANEMOI_DATASET_CACHE_PROFILE"] = "1"
    try:
        cached = _run_training(config, cache_root)
    finally:
        if previous_profile_setting is None:
            os.environ.pop("ANEMOI_DATASET_CACHE_PROFILE")
        else:
            os.environ["ANEMOI_DATASET_CACHE_PROFILE"] = previous_profile_setting

    assert len(uncached.durations) == len(cached.durations) == 3
    assert cached.cache_stats[0][2] > 0, "cold epoch did not populate the cache"
    assert all(local + remote > 0 and misses == 0 for local, remote, misses in cached.cache_stats[1:]), (
        f"cache was not warm after epoch 0: {cached.cache_stats}"
    )

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
            "cache_io_seconds(local,remote,source,store,fetch_many)=%s "
            "dataloader_wait(no_cache,cached)=%s",
            test_case,
            no_cache_seconds,
            cold_cache_seconds,
            warm_cache_seconds,
            no_cache_seconds / cold_cache_seconds,
            no_cache_seconds / warm_cache_seconds,
            cached.cache_stats,
            cached.cache_io_seconds,
            (uncached.dataloader_wait_seconds, cached.dataloader_wait_seconds),
        )
        shutil.rmtree(cache_root)
    if dist.is_initialized():
        dist.barrier()