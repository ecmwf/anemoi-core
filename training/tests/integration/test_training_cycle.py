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
from pathlib import Path

import pytest
from omegaconf import DictConfig

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.train.profiler import AnemoiProfiler
from anemoi.utils.testing import skip_if_offline

from torch.cuda import memory_stats, reset_peak_memory_stats

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_architecture_configs(architecture_config_with_data: DictConfig) -> None:
    AnemoiTrainer(architecture_config_with_data).train()


def test_config_validation_architecture_configs(architecture_config: DictConfig) -> None:
    BaseSchema(**architecture_config)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_stretched(stretched_config_with_data: DictConfig) -> None:
    AnemoiTrainer(stretched_config_with_data).train()


def test_config_validation_stretched(stretched_config: DictConfig) -> None:
    BaseSchema(**stretched_config)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_lam(lam_config_with_data: DictConfig) -> None:
    AnemoiTrainer(lam_config_with_data).train()


def test_config_validation_lam(lam_config: DictConfig) -> None:
    BaseSchema(**lam_config)


@skip_if_offline
@pytest.mark.longtests
def test_training_cycle_ensemble(ensemble_config_with_data: DictConfig) -> None:
    AnemoiTrainer(ensemble_config_with_data).train()


def test_config_validation_ensemble(ensemble_config: DictConfig) -> None:
    BaseSchema(**ensemble_config)

#return_val = value for speed profiler or 'avg_time' for time_profiler
def open_log_file(filename):
    import os
    import glob
    import csv
    if filename == "time_profiler.csv":
        return_val="avg_time"
        row_selector="name"
        row_name="run_training_batch"
    elif filename == "speed_profiler.csv":
        return_val="value"
        row_selector="metric"
        row_name="training_avg_throughput"
    else:
        raise ValueError
    tmpdir=os.getenv("TMPDIR")
    user=os.getenv("USER") #TODO should use a more portable and secure way
    file_path = next(iter(glob.glob(f"{tmpdir}/pytest-of-{user}/pytest-0/test_benchmark_training_cycle0profiler/[a-z0-9]*/{filename}")))
    with Path(file_path).open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get(row_selector) == row_name:
                result=row.get(return_val)
                break
    return float(result)
                

#@skip_if_offline
#add flag to save snapshot
#add multi-gpu support
#read from database on the s3 bucket, of a csv
#complain if any of the values differ by more than 10%
# can skip multi-gpu
# add compute/nccl/memory breakdown from pytorch profiler
@pytest.mark.longtests
def test_benchmark_training_cycle(benchmark_config_with_data: DictConfig) -> None:
    reset_peak_memory_stats()
    AnemoiProfiler(benchmark_config_with_data).profile()
    
    #read memory and mlflow stats
    stats=memory_stats(device=0)
    peak_active_mem_mb=stats['active_bytes.all.peak']/1024/1024 
    av_training_throughput = open_log_file("speed_profiler.csv")
    av_training_batch_time_s = open_log_file("time_profiler.csv")
    
    print(f"Peak memory: {peak_active_mem_mb:.2f}MB")
    print(f"Av. training batch time: {av_training_batch_time_s}s")
    print(f"Av. training throughput: {av_training_throughput}iter/s")

@skip_if_offline
@pytest.mark.longtests
def test_restart_training(gnn_config_with_data: DictConfig) -> None:

    AnemoiTrainer(gnn_config_with_data).train()

    cfg = gnn_config_with_data
    output_dir = Path(cfg.hardware.paths.output + "checkpoint")

    assert output_dir.exists(), f"Checkpoint directory not found at: {output_dir}"

    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert (
        len(run_dirs) == 1
    ), f"Expected exactly one run_id directory, found {len(run_dirs)}: {[d.name for d in run_dirs]}"

    checkpoint_dir = run_dirs[0]
    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 2, "Expected 2 checkpoints after first run"

    cfg.training.run_id = checkpoint_dir.name
    cfg.training.max_epochs = 3
    AnemoiTrainer(cfg).train()

    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 3, "Expected 3 checkpoints after second run"
