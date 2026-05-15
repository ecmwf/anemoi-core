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
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats
from torch_geometric.data import HeteroData

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.create import load_graph_from_file
from anemoi.graphs.create import validate_loaded_graph
from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.utils.config import get_multiple_datasets_config
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
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase,
) -> None:
    """Runs a benchmark for dataloader performance, testing MultiDataset batch sampling speed."""
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    cfg, test_case = benchmark_config

    original_seed, random_seed = set_temp_base_seed()
    LOGGER.info("Benchmarking dataloader for configuration: %s (seed=%s)", test_case, random_seed)

    try:
        # Initialize task
        task = instantiate(cfg.task)

        # Initialize datamodule
        datamodule = AnemoiDatasetsDataModule(config=cfg, task=task)

        # Build graph_data from the config (mirrors AnemoiTrainer.graph_data logic)
        dataset_names = list(get_multiple_datasets_config(cfg.dataloader.training).keys())
        graph_cfg = cfg.graph
        graph_path = cfg.system.input.graph
        save_path = Path(graph_path) if graph_path else None
        graph_config = OmegaConf.create(OmegaConf.to_container(graph_cfg, resolve=False))

        if save_path and save_path.exists() and not graph_cfg.get("overwrite", False):
            fused = uses_fused_dataset_graph(graph_cfg, dataset_names)
            required = dataset_names if fused else [DEFAULT_DATASET_NAME]
            graph_data = load_graph_from_file(save_path)
            validate_loaded_graph(graph_data, required)
            LOGGER.info("Loaded graph from %s", save_path)
        else:
            graph_data = GraphCreator(graph_config).create(save_path=save_path, overwrite=graph_cfg.get("overwrite", False))
            LOGGER.info("Built graph from config")

        # Compute shard_sizes: dict[dataset_name -> list[int]] using graph node counts
        reader_group_size = int(cfg.system.hardware.num_gpus_per_model)
        fused = uses_fused_dataset_graph(graph_data, dataset_names)
        shard_sizes = {}
        for name in dataset_names:
            node_key = name if fused else DEFAULT_DATASET_NAME
            grid_size = graph_data[node_key].num_nodes
            shard_sizes[name] = get_balanced_partition_sizes(grid_size, reader_group_size)
        LOGGER.info("Computed shard_sizes: %s", {k: v for k, v in shard_sizes.items()})

        # Get training dataloader
        train_dataloader = datamodule.train_dataloader()

        train_dataloader.dataset.set_comm_group_info(
            global_rank=0,
            model_comm_group_id=0,
            model_comm_group_rank=0,
            model_comm_num_groups=1,
            reader_group_rank=0,
            reader_group_size=reader_group_size,
            shard_sizes=shard_sizes,
        )
        LOGGER.info("Initialized training dataloader with batch size %d, reader_group_size: %d", train_dataloader.batch_size, train_dataloader.dataset.reader_group_size)

        rss_before = get_tree_rss_mib()
        LOGGER.info("Process tree RSS before benchmark: %.2f MiB", rss_before)


        num_warmup_batches = 10
        LOGGER.info("Warming up with %d batches", num_warmup_batches)
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= num_warmup_batches:
                break

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
        #track_dataloader_benchmark_results(test_case, batches_per_second)
    finally:
        restore_base_seed(original_seed)


@pytest.mark.multigpu
@pytest.mark.slow
def test_benchmark_training_cycle(
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase, task
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
