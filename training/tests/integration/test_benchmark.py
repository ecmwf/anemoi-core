# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import gc
import logging
import os
import time
from pathlib import Path

import psutil
import pytest
from omegaconf import DictConfig
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats

from anemoi.training.diagnostics.benchmark_server import benchmark
from anemoi.training.diagnostics.benchmark_server import parse_benchmark_config
from anemoi.training.diagnostics.benchmark_server import track_dataloader_benchmark_results
from anemoi.training.train.profiler import AnemoiProfiler

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce memory fragmentation

LOGGER = logging.getLogger(__name__)


# Record total process tree RSS before the benchmark
def get_tree_rss_mib() -> float:
    """Sum RSS of current process and all children (in MiB)."""
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total += child.memory_info().rss
    return total / (1024 * 1024)


def set_temp_base_seed() -> tuple[str | None, str]:
    """Set a temporary time-based seed and return original/new values."""
    original_seed = os.environ.get("ANEMOI_BASE_SEED")
    random_seed = str(int(time.time()))
    os.environ["ANEMOI_BASE_SEED"] = random_seed
    return original_seed, random_seed


def restore_base_seed(original_seed: str | None) -> None:
    """Restore ANEMOI_BASE_SEED to its previous value."""
    if original_seed is None:
        os.environ.pop("ANEMOI_BASE_SEED", None)
    else:
        os.environ["ANEMOI_BASE_SEED"] = original_seed


@pytest.mark.multigpu
@pytest.mark.slow
def test_benchmark_dataloader(
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase
) -> None:
    """Runs a benchmark for dataloader performance, testing MultiDataset batch sampling speed."""
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    cfg, test_case = benchmark_config

    original_seed, random_seed = set_temp_base_seed()
    LOGGER.info("Benchmarking dataloader for configuration: %s (seed=%s)", test_case, random_seed)

    try:
        # Initialize datamodule
        datamodule = AnemoiDatasetsDataModule(config=cfg)

        # Get training dataloader
        train_dataloader = datamodule.train_dataloader()

        rss_before = get_tree_rss_mib()
        LOGGER.info("Process tree RSS before benchmark: %.2f MiB", rss_before)

        # Benchmark batch sampling speed
        num_batches_to_test = 100
        LOGGER.info("Testing %d batches from MultiDataset", num_batches_to_test)

        start_time = time.perf_counter()
        batch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= num_batches_to_test:
                break
            batch_count += 1

            # Log first batch structure
            if batch_idx == 0:
                LOGGER.info("First batch structure:")
                for dataset_name, data in batch.items():
                    size_mb = data.nelement() * data.element_size() / (1024 * 1024)
                    LOGGER.info(
                        "  Dataset '%s': shape %s, dtype %s, size %.2f MB",
                        dataset_name,
                        data.shape,
                        data.dtype,
                        size_mb,
                    )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Calculate performance metrics
        batches_per_second = batch_count / elapsed_time
        time_per_batch_ms = (elapsed_time / batch_count) * 1000

        # Record total process tree RSS after the benchmark
        rss_after = get_tree_rss_mib()

        LOGGER.info("Dataloader Performance Results:")
        LOGGER.info("  Total batches: %d", batch_count)
        LOGGER.info("  Total time: %.2f seconds", elapsed_time)
        LOGGER.info("  Throughput: %.2f it/s", batches_per_second)
        LOGGER.info("  Time per batch: %.2f ms", time_per_batch_ms)
        LOGGER.info("  Process tree RSS before: %.2f MiB", rss_before)
        LOGGER.info("  Process tree RSS after:  %.2f MiB", rss_after)
        LOGGER.info("  Process tree RSS delta:  %.2f MiB", rss_after - rss_before)
        track_dataloader_benchmark_results(test_case, batches_per_second)
    finally:
        restore_base_seed(original_seed)


@pytest.mark.multigpu
@pytest.mark.slow
def test_benchmark_training_cycle(
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase
) -> None:
    """Runs a benchmark and then compares them against the values stored on a server."""
    cfg, test_case = benchmark_config
    LOGGER.info("Benchmarking the configuration: %s", test_case)

    # Reset memory logging and free all possible memory between runs
    # this ensures we report the peak memory used during each run,
    # and not the peak memory used by the run with the highest memory usage
    reset_peak_memory_stats()
    empty_cache()
    gc.collect()
    # Run model with profiler
    AnemoiProfiler(cfg).profile()

    # determine store from benchmark config
    config_path = Path("~/.config/anemoi/anemoi-benchmark.yaml").expanduser()
    user, hostname, path = parse_benchmark_config(config_path)
    store: str = f"ssh://{user}@{hostname}:{path}"

    benchmark(cfg, test_case, store)
