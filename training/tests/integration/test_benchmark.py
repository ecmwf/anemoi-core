# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os

import pytest
from omegaconf import DictConfig
from torch.cuda import reset_peak_memory_stats

from anemoi.training.diagnostics.benchmark_server import benchmark
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

    # Run model with profiler
    reset_peak_memory_stats()
    AnemoiProfiler(cfg).profile()

    store: str = "ssh://data@anemoi.ecmwf.int:/home/data/public/anemoi-integration-tests/training/benchmarks"
    benchmark(cfg, test_case, store, throw_error=True)
