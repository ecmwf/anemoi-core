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

import numpy as np
import pandas as pd

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import ValidationError
from schemas.partial_metadata_schema import PARTIAL_METADATA_SCHEMA

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.utils.mlflow.client import AnemoiMlflowClient
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline
from anemoi.training.utils.config import load_config

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners


LOGGER = logging.getLogger(__name__)


def assert_keys_exist(data: dict, schema: dict, path: str = "root") -> None:
    """Recursively check that the metadata dictionary conforms to the expected schema.

    This is a simplified schema validation that only checks for the presence of expected keys.
    Note that this does not ensure that changes in anemoi-core do not break anemoi-inference.
    """
    for key, subschema in schema.items():
        if key == "__datasets__":
            dataset_names = data.get("dataset_names", [])
            for ds in dataset_names:
                assert ds in data, f"{path}: dataset '{ds}' missing"
                assert_keys_exist(data[ds], subschema, f"{path}.{ds}")
            continue

        assert key in data, f"{path}: missing key '{key}'"

        if isinstance(subschema, dict):
            assert isinstance(data[key], dict), f"{path}.{key} should be dict"
            assert_keys_exist(data[key], subschema, f"{path}.{key}")

        if subschema is list:
            assert isinstance(data[key], list), f"{path}.{key} should be list"


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_global(
    global_config: tuple[DictConfig, str, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url, _ = global_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_global_config(global_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = global_config
    BaseSchema(**cfg)


def test_config_validation_rejects_invalid_projection_kind(global_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = global_config
    cfg.diagnostics.plot.projection_kind = "invalid_projection"
    with pytest.raises(ValidationError, match="projection_kind"):
        BaseSchema(**cfg)


def test_config_without_validation_accepts_invalid_projection_kind(global_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = global_config
    cfg.config_validation = False
    cfg.diagnostics.plot.projection_kind = "invalid_projection"
    cfg_obj = OmegaConf.to_object(cfg)
    unvalidated = UnvalidatedBaseSchema(**DictConfig(cfg_obj))
    assert unvalidated.diagnostics.plot.projection_kind == "invalid_projection"


def test_config_validation_mlflow_configs(gnn_config_mlflow: DictConfig) -> None:
    from anemoi.training.diagnostics.logger import get_mlflow_logger
    from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

    config = gnn_config_mlflow
    if config.config_validation:
        OmegaConf.resolve(config)
        config = BaseSchema(**config)
        assert config.diagnostics.log.mlflow.target_ == "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"
    else:
        config = OmegaConf.to_object(config)
        config = UnvalidatedBaseSchema(**DictConfig(config))

    from anemoi.training.schemas.base_schema import convert_to_omegaconf

    config = convert_to_omegaconf(config)

    # Minimal inputs required by get_mlflow_logger
    run_id = None
    fork_run_id = None
    paths = config.system.output
    logger_config = config.diagnostics.log

    logger_cfg = getattr(logger_config, "mlflow", None)
    if getattr(logger_cfg, "enabled", False):
        LOGGER.info("%s logger enabled", "MLFLOW")

        logger = get_mlflow_logger(
            run_id=run_id,
            fork_run_id=fork_run_id,
            paths=paths,
            logger_config=logger_config,
        )
        assert Path(logger_config.mlflow.save_dir) == Path(config.system.output.logs.mlflow)
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
    cfg.system.input.graph = "dummpy.pt"  # Mandatory input when running without config validation
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
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_stretched(stretched_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = stretched_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_multidatasets(
    multidatasets_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = multidatasets_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_multidatasets(multidatasets_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = multidatasets_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_lam(lam_config: tuple[DictConfig, list[str]], get_test_archive: GetTestArchive) -> None:
    cfg, urls = lam_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


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
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


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
def test_training_cycle_autoencoder(
    autoencoder_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, urls = autoencoder_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_autoencoder(autoencoder_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = autoencoder_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_restart_training(gnn_config: tuple[DictConfig, str], get_test_archive: GetTestArchive) -> None:
    cfg, url = gnn_config
    get_test_archive(url)

    AnemoiTrainer(cfg).train()
    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)

    assert output_dir.exists(), f"Checkpoint directory not found at: {output_dir}"

    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert len(run_dirs) == 1, (
        f"Expected exactly one run_id directory, found {len(run_dirs)}: {[d.name for d in run_dirs]}"
    )

    checkpoint_dir = run_dirs[0]
    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 2, "Expected 2 checkpoints after first run"

    cfg.training.run_id = checkpoint_dir.name
    cfg.training.max_epochs = 3
    trainer = AnemoiTrainer(cfg)
    trainer.train()

    expected_global_step = int(cfg.training.max_epochs * cfg.dataloader.limit_batches.training)
    assert trainer.model.trainer.global_step == expected_global_step, (
        f"Expected global_step={expected_global_step}, got {trainer.model.trainer.global_step}"
    )

    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 3, "Expected 3 checkpoints after second run"


@skip_if_offline
def test_loading_checkpoint(
    global_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: callable,
) -> None:
    cfg, url = global_config_with_checkpoint
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.model


@skip_if_offline
@pytest.mark.slow
def test_restart_from_existing_checkpoint(
    global_config_with_checkpoint: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = global_config_with_checkpoint
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
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_interpolator(interpolator_config: tuple[DictConfig, str]) -> None:
    """Schema-level validation for the temporal interpolation config."""
    cfg, _ = interpolator_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_diffusion(diffusion_config: tuple[DictConfig, str], get_test_archive: callable) -> None:
    cfg, url = diffusion_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_diffusion(diffusion_config: tuple[DictConfig, str]) -> None:
    cfg, _ = diffusion_config
    BaseSchema(**cfg)


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
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_imerg_target(
    imerg_target_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = imerg_target_config
    get_test_archive(url)
    AnemoiTrainer(cfg).train()


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_multidatasets_diffusion(
    multidatasets_diffusion_config: tuple[DictConfig, list[str]],
    get_test_archive: callable,
) -> None:
    cfg, urls = multidatasets_diffusion_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_build() -> None:

    config = load_config("training/tests/integration/config/atmo_integration_test.yaml")

    assert config["diagnostics"]["log"]["interval"] == 50

    trainer = AnemoiTrainer(config)
    assert trainer.cfg.diagnostics.log.interval == 50

    trainer.train()

    client = AnemoiMlflowClient("https://mlflow.ecmwf.int/", authentication=True)
    experiment = client.get_experiment_by_name("aifs-convergence")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )

    REFERENCE_ID = "e00340e8cd5c41d2881afd2265677321"

    def get_loss_df(run_id):
        history = client.get_metric_history(
            run_id,
            "train_multi_dataset_loss_step",
        )
        return pd.DataFrame(
            {
                "step": [m.step for m in history],
                "loss": [m.value for m in history],
            }
        ).set_index("step")

    def is_similar(run_id1, run_id2):
        df1, df2 = get_loss_df(run_id1), get_loss_df(run_id2)
        return np.allclose(df1.loc[:, "loss"], df2.loc[:, "loss"])

    assert is_similar(runs[0].info.run_id, REFERENCE_ID), f"Loss curve for run {runs[0].info.run_id} does not match reference"