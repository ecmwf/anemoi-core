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

from anemoi.training.api import normalize_config
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_architecture_configs(
    architecture_config: tuple[DictConfig, str, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url, _ = architecture_config
    get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_architecture_configs(architecture_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = architecture_config
    normalize_config(cfg)


def test_normalize_config_mlflow_configs(base_global_config: tuple[DictConfig, str, str]) -> None:
    from anemoi.training.diagnostics.logger import get_mlflow_logger
    from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

    config, _, _ = base_global_config
    config = normalize_config(config)
    assert config.diagnostics.log.mlflow._target_ == "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"

    logger = get_mlflow_logger(config)

    if config.diagnostics.log.mlflow.enabled:
        assert Path(config.diagnostics.log.mlflow.save_dir) == Path(config.system.output.logs.mlflow)
        assert isinstance(logger, AnemoiMLflowLogger)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_stretched(
    stretched_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = stretched_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_stretched(stretched_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = stretched_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_multidatasets(
    multidatasets_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = multidatasets_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_multidatasets(multidatasets_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = multidatasets_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_lam(lam_config: tuple[DictConfig, list[str]], get_test_archive: GetTestArchive) -> None:
    cfg, urls = lam_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_lam_with_existing_graph(
    lam_config_with_graph: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = lam_config_with_graph
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_lam(lam_config: DictConfig) -> None:
    cfg, _ = lam_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_ensemble(ensemble_config: tuple[DictConfig, str], get_test_archive: GetTestArchive) -> None:
    cfg, url = ensemble_config
    get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_ensemble(ensemble_config: tuple[DictConfig, str]) -> None:
    cfg, _ = ensemble_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_hierarchical(
    hierarchical_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = hierarchical_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_hierarchical(hierarchical_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = hierarchical_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
def test_restart_training(gnn_config: tuple[DictConfig, str], get_test_archive: GetTestArchive) -> None:
    cfg, url = gnn_config
    get_test_archive(url)

    trainer = AnemoiTrainer(normalize_config(cfg))
    trainer.train()
    checkpoint_root = Path(trainer.config.system.output.checkpoints.root)
    run_dir = checkpoint_root

    assert run_dir.exists(), f"Checkpoint directory not found at: {run_dir}"
    assert len(list(run_dir.glob("anemoi-by_epoch-*.ckpt"))) == 2, "Expected 2 checkpoints after first run"

    cfg.training.run_id = run_dir.name
    cfg.training.max_epochs = 3
    trainer = AnemoiTrainer(normalize_config(cfg))
    trainer.train()

    assert trainer.model.trainer.global_step == 6

    assert len(list(run_dir.glob("anemoi-by_epoch-*.ckpt"))) == 3, "Expected 3 checkpoints after second run"


@skip_if_offline
def test_loading_checkpoint(
    architecture_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: callable,
) -> None:
    cfg, url = architecture_config_with_checkpoint
    get_test_archive(url)
    trainer = AnemoiTrainer(normalize_config(cfg))
    trainer.model


@skip_if_offline
@pytest.mark.slow
def test_restart_from_existing_checkpoint(
    architecture_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = architecture_config_with_checkpoint
    get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_interpolator(
    interpolator_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    """Full training-cycle smoke-test for the temporal interpolation task."""
    cfg, url = interpolator_config
    get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_interpolator(interpolator_config: tuple[DictConfig, str]) -> None:
    """Adapter-level validation for the temporal interpolation config."""
    cfg, _ = interpolator_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_diffusion(diffusion_config: tuple[DictConfig, str], get_test_archive: callable) -> None:
    cfg, url = diffusion_config
    get_test_archive(url)
    AnemoiTrainer(normalize_config(cfg)).train()


def test_normalize_config_diffusion(diffusion_config: tuple[DictConfig, str]) -> None:
    cfg, _ = diffusion_config
    normalize_config(cfg)


@skip_if_offline
@pytest.mark.slow
@pytest.mark.mlflow
def test_training_cycle_mlflow_dry_run(
    mlflow_dry_run_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    from anemoi.training.commands.mlflow import prepare_mlflow_run_id

    cfg, url = mlflow_dry_run_config

    # Generate a dry run ID and set it in the config
    run_id, _ = prepare_mlflow_run_id(
        config=cfg,
    )
    cfg["training"]["run_id"] = run_id

    # Get training data
    get_test_archive(url)

    # Run training
    AnemoiTrainer(normalize_config(cfg)).train()
