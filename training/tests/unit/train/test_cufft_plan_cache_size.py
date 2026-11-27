# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The trainer sets the cuFFT plan cache limit of every visible GPU from training.cufft_plan_cache_size."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from pydantic import ValidationError

from anemoi.training.train.train import AnemoiTrainer


def _limits() -> list[int]:
    return [torch.backends.cuda.cufft_plan_cache[device].max_size for device in range(torch.cuda.device_count())]


@pytest.fixture(autouse=True)
def _restore_limits() -> Iterator[None]:
    previous = _limits()
    yield
    for device, size in enumerate(previous):
        torch.backends.cuda.cufft_plan_cache[device].max_size = size


def _compose_config(tmp_path: Path, extra_overrides: list[str]) -> DictConfig:
    """Return the default preset with placeholder values for its mandatory fields."""
    overrides = [
        f"system.input.dataset={tmp_path / 'dataset.zarr'}",
        f"system.input.graph={tmp_path / 'graph.pt'}",
        f"system.output.root={tmp_path / 'output'}",
        "diagnostics.log.mlflow.tracking_uri=http://localhost:5000",
        *extra_overrides,
    ]
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        return compose(config_name="config", overrides=overrides)


def _build_trainer(config: DictConfig) -> AnemoiTrainer:
    with (
        patch("anemoi.training.train.train.AnemoiTrainer._check_dry_run"),
        patch("anemoi.training.train.train.AnemoiTrainer._log_information"),
    ):
        return AnemoiTrainer(config=config)


@pytest.mark.parametrize("validation", ["true", "false"])
def test_an_unset_size_keeps_the_limits(tmp_path: Path, validation: str) -> None:
    for device in range(torch.cuda.device_count()):
        torch.backends.cuda.cufft_plan_cache[device].max_size = 123
    _build_trainer(_compose_config(tmp_path, [f"config_validation={validation}"]))
    assert _limits() == [123] * torch.cuda.device_count()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("validation", ["true", "false"])
def test_the_trainer_sets_the_limit_on_every_gpu(tmp_path: Path, validation: str) -> None:
    overrides = [f"config_validation={validation}", "+training.cufft_plan_cache_size=65536"]
    _build_trainer(_compose_config(tmp_path, overrides))
    assert _limits() == [65536] * torch.cuda.device_count()


@pytest.mark.parametrize("size", [0, -1])
def test_the_schema_rejects_a_size_below_one(tmp_path: Path, size: int) -> None:
    with pytest.raises(ValidationError, match="cufft_plan_cache_size"):
        _build_trainer(_compose_config(tmp_path, [f"+training.cufft_plan_cache_size={size}"]))
