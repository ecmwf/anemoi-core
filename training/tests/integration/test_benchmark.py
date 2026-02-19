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
from pathlib import Path

import pytest
from omegaconf import DictConfig
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats

from anemoi.training.diagnostics.benchmark_server import benchmark
from anemoi.training.diagnostics.benchmark_server import parse_benchmark_config
from anemoi.training.train.profiler import AnemoiProfiler

from anemoi.training.utils.dataset_cache import DatasetCache
import torch.distributed as dist
from torch import device

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce memory fragmentation

LOGGER = logging.getLogger(__name__)


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


def _init_parallel() -> None:
    """Initializes the distributed process group for benchmarking."""
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    #dist.init_process_group(backend="nccl", device_id=device(f"cuda:{local_rank}"))
    dist.init_process_group(backend="gloo", device_id=device(f"cuda:{local_rank}"))
    LOGGER.info("Initialized distributed process group for benchmarking: global_rank=%d, world_size=%d, local_rank=%d", global_rank, world_size, local_rank)

@pytest.mark.slow
@pytest.mark.multigpu
#@pytest.mark.parametrize("cache_dir", [None, os.getenv("TMPDIR")], ids=["no_cache", "with_cache"])
@pytest.mark.parametrize("cache_dir", [ os.getenv("TMPDIR")], ids=[ "with_cache"])
def test_benchmark_dataloader(
    benchmark_config: tuple[DictConfig, str],  # cfg, benchmarkTestCase
    cache_dir: str | None,
) -> None:
    """Runs a benchmark for dataloader performance, testing MultiDataset batch sampling speed."""
    import time

    from anemoi.graphs.create import GraphCreator
    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule

    cfg, test_case = benchmark_config
    cfg.graph.nodes.data.node_builder.dataset = cfg.system.input.dataset
    LOGGER.info("Benchmarking dataloader for configuration: %s", test_case)

    # Reset memory logging and free all possible memory between runs
    # this ensures we report the peak memory used during each run,
    # and not the peak memory used by the run with the highest memory usage
    reset_peak_memory_stats()
    empty_cache()
    gc.collect()

    _init_parallel()

    # Initialize the forecaster to get graph data
    graph = GraphCreator(config=cfg.graph).create(overwrite=True)

    # Initialize datamodule with graph data
    datamodule = AnemoiDatasetsDataModule(config=cfg, graph_data={"data": graph})
    if cache_dir is not None:
            LOGGER.info(f"'config.system.hardware.cache_dir' given. Caching dataset under '{cache_dir}'")
            #import pdb
            #breakpoint()
            dataset_path=f"{cfg.system.input.dataset}"
            datamodule = DatasetCache(ds=datamodule, cache_root=cache_dir, dataset_path=dataset_path)
            datamodule.setup()

    # Disable shuffle for benchmarking to get consistent batches across epochs
    datamodule.ds_train.shuffle = False
    LOGGER.info("Disabled shuffle in ds_train for consistent benchmarking across epochs")
    
    # Verify cache wrapper is still in place if cache is enabled
    if cache_dir is not None:
        from anemoi.training.utils.dataset_cache import CachedDataWrapper
        for dataset_name, dataset in datamodule.ds_train.datasets.items():
            is_wrapped = isinstance(dataset.data, CachedDataWrapper)
            LOGGER.info(f"Dataset '{dataset_name}' data is {'WRAPPED' if is_wrapped else 'NOT WRAPPED'} (type: {type(dataset.data).__name__})")
    
    # Get training dataloader
    train_dataloader = datamodule.train_dataloader()

    # Benchmark batch sampling speed
    num_batches_to_test = 100
    LOGGER.info("Testing %d batches per epoch, %d epochs from MultiDataset", num_batches_to_test, 2)

    if cache_dir is not None and not datamodule.is_initalised:
         raise ValueError(f"DatasetCache was not properly initialized with cache_dir '{cache_dir}'")

    for epoch_idx in range(2):
        start_time = time.perf_counter()
        batch_count = 0

        dl_iter=iter(train_dataloader)

        #for batch_idx, batch in enumerate(train_dataloader):
        for batch_idx in range(num_batches_to_test):

            if batch_idx >= num_batches_to_test:
                break
            batch_count += 1
            batch = next(dl_iter)

            # Log first batch structure
            if batch_idx == 0 and epoch_idx == 0:
                LOGGER.info("First batch structure:")
                for dataset_name, data in batch.items():
                    LOGGER.info("  Dataset '%s': shape %s, dtype %s", dataset_name, data.shape, data.dtype)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Calculate performance metrics
        batches_per_second = batch_count / elapsed_time
        time_per_batch_ms = (elapsed_time / batch_count) * 1000

        LOGGER.info("Dataloader Performance Results (epoch %d):", epoch_idx)
        LOGGER.info("  Total batches: %d", batch_count)
        LOGGER.info("  Total time: %.2f seconds", elapsed_time)
        LOGGER.info("  Throughput: %.2f it/s", batches_per_second)
        LOGGER.info("  Time per batch: %.2f ms", time_per_batch_ms)
        
        # Print cache statistics if cache is enabled
        if cache_dir is not None and hasattr(datamodule, 'print_cache_stats'):
            datamodule.print_cache_stats()
            
            # After first epoch, update global view so ranks know about each other's caches
            if epoch_idx == 0:
                LOGGER.info("Updating global cache registry after epoch 0...")
                datamodule.update_global_view()
                LOGGER.info("Global cache registry updated. Second epoch should benefit from shared cache awareness.")

    datamodule._shutdown_server()