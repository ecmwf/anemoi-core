# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The trainer's config stays valid after the run identity is resolved.

The config the trainer holds after construction is the one saved into every
checkpoint (``hyper_parameters.config`` and ``metadata.config``) and logged to
MLflow. ``training.run_id`` / ``training.fork_run_id`` are rejected by the schema,
so the trainer must keep the resolved run identity off the config; otherwise a
saved config can no longer be validated or relaunched from.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.training.train.train import AnemoiTrainer
from anemoi.training.utils.jsonify import map_config_to_primitives

_RUN_SOURCE_OVERRIDES = ["training/checkpoint/source=run", "+training.checkpoint.source.run_id=parent-run"]


def _compose_config(tmp_path: Path, extra_overrides: list[str]) -> DictConfig:
    """Compose the default preset with placeholder values for its mandatory fields."""
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


@pytest.mark.parametrize("schema_cls", [BaseSchema, UnvalidatedBaseSchema])
@pytest.mark.parametrize(
    ("extra_overrides", "expected_fork_run_id"),
    [
        pytest.param([], None, id="fresh"),
        pytest.param(_RUN_SOURCE_OVERRIDES, None, id="resume"),
        pytest.param([*_RUN_SOURCE_OVERRIDES, "training.checkpoint.source.fork=true"], "parent-run", id="fork"),
    ],
)
def test_post_init_config_passes_schema_validation(
    tmp_path: Path,
    schema_cls: type,
    extra_overrides: list[str],
    expected_fork_run_id: str | None,
) -> None:
    """The config saved into a checkpoint after init re-validates, for fresh, resumed and forked runs."""
    trainer = _build_trainer(_compose_config(tmp_path, extra_overrides))

    # The run identity is resolved, but only on the trainer.
    assert trainer.run_id
    assert trainer._fork_run_id == expected_fork_run_id
    assert "run_id" not in trainer.config.training
    assert "fork_run_id" not in trainer.config.training

    # The same conversion the trainer applies for ``metadata.config``, read back as
    # ``anemoi-training config validate`` would read a saved config.
    saved_config = OmegaConf.create(map_config_to_primitives(trainer.config))
    schema_cls(**saved_config)
