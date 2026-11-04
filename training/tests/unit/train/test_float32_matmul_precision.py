# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The trainer sets the precision of float32 matrix products from training.float32_matmul_precision."""

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


@pytest.fixture(autouse=True)
def _restore_precision() -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(previous)


def _compose_config(tmp_path: Path, extra_overrides: list[str]) -> DictConfig:
    """The default preset with placeholder values for its mandatory fields."""
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
@pytest.mark.parametrize(
    ("extra_overrides", "expected"),
    [
        pytest.param([], "high", id="not set"),
        pytest.param(["+training.float32_matmul_precision=highest"], "highest", id="highest"),
    ],
)
def test_the_trainer_sets_the_configured_precision(
    tmp_path: Path,
    validation: str,
    extra_overrides: list[str],
    expected: str,
) -> None:
    torch.set_float32_matmul_precision("medium")
    _build_trainer(_compose_config(tmp_path, [f"config_validation={validation}", *extra_overrides]))
    assert torch.get_float32_matmul_precision() == expected


def test_the_schema_rejects_an_unknown_precision(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="float32_matmul_precision"):
        _build_trainer(_compose_config(tmp_path, ["+training.float32_matmul_precision=low"]))
