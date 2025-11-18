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
from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
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
    AnemoiTrainer(cfg).train()


def test_config_validation_architecture_configs(architecture_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = architecture_config
    BaseSchema(**cfg)


def test_config_validation_mlflow_configs(base_global_config: tuple[DictConfig, str, str]) -> None:
    from anemoi.training.diagnostics.logger import get_mlflow_logger
    from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

    config, _, _ = base_global_config
    if config.config_validation:
        OmegaConf.resolve(config)
        config = BaseSchema(**config)
        assert config.diagnostics.log.mlflow.target_ == "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"
    else:
        config = OmegaConf.to_object(config)
        config = UnvalidatedBaseSchema(**DictConfig(config))

    logger = get_mlflow_logger(config)

    assert Path(config.diagnostics.log.mlflow.save_dir) == Path(config.hardware.paths.logs.mlflow)
    if config.diagnostics.log.mlflow.enabled:
        assert isinstance(logger, AnemoiMLflowLogger)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_without_config_validation(
    gnn_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = gnn_config
    get_test_archive(url)

    cfg.config_validation = False
    cfg.hardware.files.graph = "dummpy.pt"  # Mandatory input when running without config validation
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_stretched(
    stretched_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = stretched_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(cfg).train()


def test_config_validation_stretched(stretched_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = stretched_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_lam(lam_config: tuple[DictConfig, list[str]], get_test_archive: GetTestArchive) -> None:
    cfg, urls = lam_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_lam_with_existing_graph(
    lam_config_with_graph: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = lam_config_with_graph
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(cfg).train()


def test_config_validation_lam(lam_config: DictConfig) -> None:
    cfg, _ = lam_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_ensemble(ensemble_config: tuple[DictConfig, str], get_test_archive: GetTestArchive) -> None:
    cfg, url = ensemble_config
    get_test_archive(url)
    AnemoiTrainer(cfg).train()


def test_config_validation_ensemble(ensemble_config: tuple[DictConfig, str]) -> None:
    cfg, _ = ensemble_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_hierarchical(
    hierarchical_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = hierarchical_config
    for url in urls:
        get_test_archive(url)
    AnemoiTrainer(cfg).train()


def test_config_validation_hierarchical(hierarchical_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = hierarchical_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_restart_training(gnn_config: tuple[DictConfig, str], get_test_archive: GetTestArchive) -> None:
    cfg, url = gnn_config
    get_test_archive(url)

    AnemoiTrainer(cfg).train()

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


@skip_if_offline
def test_loading_checkpoint(
    architecture_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: callable,
) -> None:
    cfg, url = architecture_config_with_checkpoint
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.model


@skip_if_offline
@pytest.mark.slow
def test_restart_from_existing_checkpoint(
    architecture_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = architecture_config_with_checkpoint
    get_test_archive(url)
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_interpolator(
    interpolator_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    """Full training-cycle smoke-test for the temporal interpolation task."""
    cfg, url = interpolator_config
    get_test_archive(url)
    AnemoiTrainer(cfg).train()


def test_config_validation_interpolator(interpolator_config: tuple[DictConfig, str]) -> None:
    """Schema-level validation for the temporal interpolation config."""
    cfg, _ = interpolator_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_diffusion(diffusion_config: tuple[DictConfig, str], get_test_archive: callable) -> None:
    cfg, url = diffusion_config
    get_test_archive(url)
    AnemoiTrainer(cfg).train()


def test_config_validation_diffusion(diffusion_config: tuple[DictConfig, str]) -> None:
    cfg, _ = diffusion_config
    BaseSchema(**cfg)
