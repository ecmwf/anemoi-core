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
    config_path = Path("~/.config/anemoi-benchmark.yaml").expanduser()
    user, hostname, path = parse_benchmark_config(config_path)
    store: str = f"ssh://{user}@{hostname}:{path}"

    benchmark(cfg, test_case, store)
