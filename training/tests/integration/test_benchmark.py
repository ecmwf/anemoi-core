# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import gc
import logging
import os
import resource
from pathlib import Path

import pytest
from omegaconf import DictConfig
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats

from anemoi.training.diagnostics.benchmark_server import benchmark
from anemoi.training.diagnostics.benchmark_server import parse_benchmark_config
from anemoi.training.train.profiler import AnemoiProfiler

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce memory fragmentation

LOGGER = logging.getLogger(__name__)


@pytest.mark.multigpu
@pytest.mark.slow
def test_benchmark_dataloader(
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase
) -> None:
    """Runs a benchmark for dataloader performance, testing MultiDataset batch sampling speed."""
    import time

    from anemoi.graphs.create import GraphCreator
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    cfg, test_case = benchmark_config
    cfg.graph.nodes.data.node_builder.dataset = cfg.system.input.dataset
    LOGGER.info("Benchmarking dataloader for configuration: %s", test_case)

    # Initialize the forecaster to get graph data
    graph = GraphCreator(config=cfg.graph).create(overwrite=True)

    # Initialize datamodule with graph data
    datamodule = AnemoiDatasetsDataModule(config=cfg, graph_data={"data": graph})

    # Get training dataloader
    train_dataloader = datamodule.train_dataloader()

    # Record current CPU memory (RSS) for the whole process tree before the benchmark
    def _read_rss_kib(pid: int) -> int:
        """Read VmRSS (in kB) from /proc/<pid>/status. Returns 0 on failure."""
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        return 0

    def _get_descendant_pids(root_pid: int) -> list[int]:
        """Return all descendant PIDs of root_pid by walking /proc/*/status."""
        descendants = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                            if ppid == root_pid:
                                child_pid = int(entry)
                                descendants.append(child_pid)
                                descendants.extend(_get_descendant_pids(child_pid))
                            break
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return descendants

    def get_process_tree_rss_kib() -> tuple[int, int, int]:
        """Return (master_rss, children_rss, total_rss) in kB for the whole process tree."""
        my_pid = os.getpid()
        master_rss = _read_rss_kib(my_pid)
        children_rss = sum(_read_rss_kib(pid) for pid in _get_descendant_pids(my_pid))
        return master_rss, children_rss, master_rss + children_rss

    master_before, children_before, total_before = get_process_tree_rss_kib()
    LOGGER.info(
        "CPU RSS before benchmark: master %.2f MiB, children %.2f MiB, total %.2f MiB",
        master_before / 1024, children_before / 1024, total_before / 1024,
    )

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
                LOGGER.info("  Dataset '%s': shape %s, dtype %s, size %.2f MB", dataset_name, data.shape, data.dtype, size_mb)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Calculate performance metrics
    batches_per_second = batch_count / elapsed_time
    time_per_batch_ms = (elapsed_time / batch_count) * 1000

    # Record current CPU memory (RSS) for the whole process tree after the benchmark
    master_after, children_after, total_after = get_process_tree_rss_kib()

    LOGGER.info("Dataloader Performance Results:")
    LOGGER.info("  Total batches: %d", batch_count)
    LOGGER.info("  Total time: %.2f seconds", elapsed_time)
    LOGGER.info("  Throughput: %.2f it/s", batches_per_second)
    LOGGER.info("  Time per batch: %.2f ms", time_per_batch_ms)
    LOGGER.info("  Master RSS before:   %.2f MiB", master_before / 1024)
    LOGGER.info("  Master RSS after:    %.2f MiB", master_after / 1024)
    LOGGER.info("  Master RSS delta:    %.2f MiB", (master_after - master_before) / 1024)
    LOGGER.info("  Children RSS before: %.2f MiB", children_before / 1024)
    LOGGER.info("  Children RSS after:  %.2f MiB", children_after / 1024)
    LOGGER.info("  Children RSS delta:  %.2f MiB", (children_after - children_before) / 1024)
    LOGGER.info("  Total RSS before:    %.2f MiB", total_before / 1024)
    LOGGER.info("  Total RSS after:     %.2f MiB", total_after / 1024)
    LOGGER.info("  Total RSS delta:     %.2f MiB", (total_after - total_before) / 1024)


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
