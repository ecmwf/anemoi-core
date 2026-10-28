# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""How the trainer resolves and applies the run identity.

The output paths follow the run (or fork, or server-to-server parent) id, the
dry-run decision is the same on every rank, and a keyless run gets a fresh uuid4.

The trainer's config also stays valid after the run identity is resolved. The
config the trainer holds after construction is the one saved into every
checkpoint (``hyper_parameters.config`` and ``metadata.config``) and logged to
MLflow. ``training.run_id`` / ``training.fork_run_id`` are rejected by the schema,
so the trainer must keep the resolved run identity off the config; otherwise a
saved config can no longer be validated or relaunched from.
"""

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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


# --- output paths follow the run identity --------------------------------------------


def _paths_namespace(
    *,
    run_id: str | None,
    parent_run_server2server: str | None,
    fork_run_id: str | None,
) -> SimpleNamespace:
    """Minimal ``self`` for ``AnemoiTrainer._update_paths``."""
    config = OmegaConf.create(
        {
            "system": {"output": {"checkpoints": {"root": "/base/ckpts"}, "plots": "/base/plots"}},
            "training": {},
        },
    )
    return SimpleNamespace(
        run_id=run_id,
        parent_run_server2server=parent_run_server2server,
        _run_identity=(run_id, fork_run_id),
        _resume_run_id=run_id,
        _fork_run_id=fork_run_id,
        config=config,
    )


def test_update_paths_appends_the_run_id_to_checkpoints_and_plots() -> None:
    ns = _paths_namespace(run_id="run-abc", parent_run_server2server=None, fork_run_id=None)

    AnemoiTrainer._update_paths(ns)

    assert ns.config.system.output.checkpoints.root == Path("/base/ckpts", "run-abc")
    assert ns.config.system.output.plots == Path("/base/plots", "run-abc")


def test_update_paths_uses_the_fork_run_id_for_checkpoints_only() -> None:
    ns = _paths_namespace(run_id=None, parent_run_server2server=None, fork_run_id="fork-999")

    AnemoiTrainer._update_paths(ns)

    assert ns.config.system.output.checkpoints.root == Path("/base/ckpts", "fork-999")
    assert ns.config.system.output.plots == "/base/plots"


def test_update_paths_prefers_the_server2server_parent_over_the_run_id() -> None:
    ns = _paths_namespace(run_id="local-run", parent_run_server2server="remote-parent", fork_run_id=None)

    AnemoiTrainer._update_paths(ns)

    assert ns.config.system.output.checkpoints.root == Path("/base/ckpts", "remote-parent")
    assert ns.config.system.output.plots == Path("/base/plots", "remote-parent")


# --- the dry-run decision -----------------------------------------------------------


def _dry_run_namespace(*, fork_run_id: str | None, parent_dry_run: bool) -> SimpleNamespace:
    """Minimal ``self`` for ``AnemoiTrainer._check_dry_run`` with an MLflow logger."""
    config = OmegaConf.create(
        {
            # A path that does not exist on disk, so ``.is_dir()`` is False.
            "system": {"output": {"checkpoints": {"root": "/nonexistent/checkpoint/dir"}}},
            "training": {},
        },
    )
    return SimpleNamespace(
        start_from_checkpoint=True,
        logger=SimpleNamespace(logger_name="mlflow"),
        mlflow_logger=SimpleNamespace(_parent_dry_run=parent_dry_run),
        _run_identity=(None, fork_run_id),
        _resume_run_id=None,
        _fork_run_id=fork_run_id,
        config=config,
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_check_dry_run_suppresses_start_from_checkpoint_for_a_dry_parent_on_every_rank(
    rank: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry parent run has no checkpoint to start from, and every rank must reach that answer.

    The decision controls whether the checkpoint pipeline acquires anything at all.
    A rank-0-only answer would have rank 0 start fresh while every other rank looked
    for a checkpoint that was never written. There is no process group yet at trainer
    construction to broadcast over, and both inputs (an MLflow tag, a directory on the
    shared filesystem) are rank-independent, so every rank computes it.
    """
    monkeypatch.setattr(rank_zero_only, "rank", rank)
    ns = _dry_run_namespace(fork_run_id=None, parent_dry_run=True)

    AnemoiTrainer._check_dry_run(ns)

    assert ns.dry_run is True
    assert ns.start_from_checkpoint is False


def test_check_dry_run_keeps_start_from_checkpoint_for_a_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fork starts from its parent's checkpoint even when the parent is a dry run."""
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    ns = _dry_run_namespace(fork_run_id="fork-1", parent_dry_run=True)

    AnemoiTrainer._check_dry_run(ns)

    assert ns.dry_run is True
    assert ns.start_from_checkpoint is True


def test_run_id_is_a_fresh_uuid4_for_a_keyless_run() -> None:
    ns = SimpleNamespace(_run_identity=(None, None), _resume_run_id=None, _fork_run_id=None, logger=None)

    run_id = AnemoiTrainer.run_id.func(ns)

    assert uuid.UUID(run_id).version == 4
    assert str(uuid.UUID(run_id)) == run_id
