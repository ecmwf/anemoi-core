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
from omegaconf import open_dict
from pydantic import ValidationError
from schemas.partial_metadata_schema import PARTIAL_METADATA_SCHEMA

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import build_schema
from anemoi.training.schemas.base_schema import ConfigValidationError
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline

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
    config = build_schema(cfg)
    assert config.diagnostics.plot.projection_kind == "invalid_projection"


def test_config_without_validation_resolves_strategy_interpolations(
    gnn_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = gnn_config
    cfg.config_validation = False
    config = build_schema(cfg)
    assert isinstance(config.training.strategy.num_gpus_per_model, int)
    assert isinstance(config.training.strategy.read_group_size, int)


def test_config_without_validation_applies_graph_builder_defaults(
    gnn_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = gnn_config
    cfg.config_validation = False

    # Ensure we verify default injection on a user-provided cutoff builder dict.
    with open_dict(cfg.graph.edges[0].edge_builders[0]):
        cfg.graph.edges[0].edge_builders[0].pop("max_num_neighbours", None)

    config = build_schema(cfg)
    edge_builder = config.graph.edges[0].edge_builders[0]
    if isinstance(edge_builder, DictConfig | dict):
        assert edge_builder.get("max_num_neighbours") == 64
    else:
        assert edge_builder.max_num_neighbours == 64


def test_config_without_validation_preserves_extra_fields(global_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = global_config
    cfg.config_validation = False
    with open_dict(cfg.training):
        cfg.training.typo_extra_field = "ignored"

    config = build_schema(cfg)

    assert config.training.typo_extra_field == "ignored"


def test_config_without_validation_preserves_invalid_field_values(
    gnn_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = gnn_config
    cfg.config_validation = False
    cfg.training.update_ds_stats_on_ckpt_load.states = {"bad": "value"}

    config = build_schema(cfg)

    assert config.training.update_ds_stats_on_ckpt_load.states == {"bad": "value"}
    assert config.training.update_ds_stats_on_ckpt_load.tendencies is True


def test_config_without_validation_requires_valid_structural_discriminator(
    gnn_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = gnn_config
    cfg.config_validation = False
    cfg.training.model_task = "not.a.real.training.task"

    with pytest.raises(ValueError, match="Cannot determine schema branch"):
        build_schema(cfg)


def test_config_validation_reports_invalid_structural_discriminator_clearly(
    gnn_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = gnn_config
    cfg.training.model_task = "not.a.real.training.task"

    with pytest.raises(ConfigValidationError, match="training.model_task"):
        build_schema(cfg)


def test_config_validation_mlflow_configs(gnn_config_mlflow: DictConfig) -> None:
    from anemoi.training.diagnostics.logger import get_mlflow_logger
    from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

    config = build_schema(gnn_config_mlflow)
    if config.config_validation:
        assert config.diagnostics.log.mlflow.target_ == "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger"

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


@pytest.mark.parametrize("config_validation", [True, False], ids=["strict_validation", "defaults_only"])
def test_pydantic_defaults_are_applied_in_both_config_modes(
    gnn_config: tuple[DictConfig, str],
    config_validation: bool,
) -> None:
    cfg, _ = gnn_config
    parsed_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(parsed_input, DictConfig)

    with open_dict(parsed_input.training):
        parsed_input.config_validation = config_validation
        parsed_input.training.pop("multistep_output", None)
        parsed_input.training.pop("update_ds_stats_on_ckpt_load", None)

    parsed_config = build_schema(parsed_input)

    assert parsed_config.training.multistep_output == 1
    assert parsed_config.training.update_ds_stats_on_ckpt_load.states is False
    assert parsed_config.training.update_ds_stats_on_ckpt_load.tendencies is True


def test_pydantic_defaults_match_between_validation_modes(gnn_config: tuple[DictConfig, str]) -> None:
    cfg, _ = gnn_config
    strict_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    defaults_only_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert isinstance(strict_input, DictConfig)
    assert isinstance(defaults_only_input, DictConfig)

    for parsed_input, config_validation in (
        (strict_input, True),
        (defaults_only_input, False),
    ):
        with open_dict(parsed_input.training):
            parsed_input.config_validation = config_validation
            parsed_input.training.pop("multistep_output", None)
            parsed_input.training.pop("update_ds_stats_on_ckpt_load", None)

    strict_config = build_schema(strict_input)
    defaults_only_config = build_schema(defaults_only_input)

    assert strict_config.training.multistep_output == defaults_only_config.training.multistep_output
    assert (
        strict_config.training.update_ds_stats_on_ckpt_load.states
        == defaults_only_config.training.update_ds_stats_on_ckpt_load.states
    )
    assert (
        strict_config.training.update_ds_stats_on_ckpt_load.tendencies
        == defaults_only_config.training.update_ds_stats_on_ckpt_load.tendencies
    )


def test_valid_config_output_matches_between_validation_modes(
    gnn_config: tuple[DictConfig, str],
) -> None:
    def _project_to_reference_shape(candidate: object, reference: object) -> object:
        if isinstance(reference, dict) and isinstance(candidate, dict):
            return {
                key: _project_to_reference_shape(candidate[key], value)
                for key, value in reference.items()
                if key in candidate
            }
        if isinstance(reference, list) and isinstance(candidate, list):
            return [
                _project_to_reference_shape(candidate_item, reference_item)
                for candidate_item, reference_item in zip(candidate, reference, strict=False)
            ]
        return candidate

    def _normalize_runtime_value(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _normalize_runtime_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_normalize_runtime_value(item) for item in value]
        return value

    cfg, _ = gnn_config
    strict_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    defaults_only_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert isinstance(strict_input, DictConfig)
    assert isinstance(defaults_only_input, DictConfig)

    strict_input.config_validation = True
    defaults_only_input.config_validation = False

    strict_config = build_schema(strict_input)
    defaults_only_config = build_schema(defaults_only_input)

    strict_output = OmegaConf.to_container(strict_config.model_dump(by_alias=True), resolve=True)
    defaults_only_output = OmegaConf.to_container(defaults_only_config.model_dump(by_alias=True), resolve=True)

    assert isinstance(strict_output, dict)
    assert isinstance(defaults_only_output, dict)

    # Compare the runtime-critical sections that should be identical between
    # strict and defaults-only modes for a valid config.
    for section in ("data", "dataloader", "system", "graph", "model", "training"):
        projected_defaults_only = _project_to_reference_shape(defaults_only_output[section], strict_output[section])
        assert _normalize_runtime_value(strict_output[section]) == _normalize_runtime_value(projected_defaults_only)


def test_config_validation_flag_only_changes_strict_semantic_checks(
    gnn_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = gnn_config
    strict_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    defaults_only_input = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert isinstance(strict_input, DictConfig)
    assert isinstance(defaults_only_input, DictConfig)

    for parsed_input, config_validation in (
        (strict_input, True),
        (defaults_only_input, False),
    ):
        with open_dict(parsed_input.training):
            parsed_input.config_validation = config_validation
            parsed_input.diagnostics.plot.projection_kind = "invalid_projection"
            parsed_input.training.pop("multistep_output", None)
            parsed_input.training.pop("update_ds_stats_on_ckpt_load", None)

    with pytest.raises(ConfigValidationError, match="diagnostics.plot.projection_kind"):
        build_schema(strict_input)

    defaults_only_config = build_schema(defaults_only_input)
    assert defaults_only_config.diagnostics.plot.projection_kind == "invalid_projection"
    assert defaults_only_config.training.multistep_output == 1
    assert defaults_only_config.training.update_ds_stats_on_ckpt_load.states is False
    assert defaults_only_config.training.update_ds_stats_on_ckpt_load.tendencies is True


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
    assert (
        len(run_dirs) == 1
    ), f"Expected exactly one run_id directory, found {len(run_dirs)}: {[d.name for d in run_dirs]}"

    checkpoint_dir = run_dirs[0]
    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 2, "Expected 2 checkpoints after first run"

    cfg.training.run_id = checkpoint_dir.name
    cfg.training.max_epochs = 3
    trainer = AnemoiTrainer(cfg)
    trainer.train()

    expected_global_step = int(cfg.training.max_epochs * cfg.dataloader.limit_batches.training)
    assert (
        trainer.model.trainer.global_step == expected_global_step
    ), f"Expected global_step={expected_global_step}, got {trainer.model.trainer.global_step}"

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
