# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from pydantic import ValidationError
from schemas.partial_metadata_schema import PARTIAL_METADATA_SCHEMA

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.training.train.evaluate import AnemoiEvaluator
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


@skip_if_offline
def test_config_validation_ensemble(ensemble_config: tuple[DictConfig, str]) -> None:
    cfg, _ = ensemble_config
    BaseSchema(**cfg)


def test_config_validation_ensemble_graph_multiscale(ensemble_graph_multiscale_config: tuple[DictConfig, str]) -> None:
    cfg, _ = ensemble_graph_multiscale_config
    BaseSchema(**cfg)


def test_config_validation_ensemble_truncated_connection(
    ensemble_truncated_connection_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = ensemble_truncated_connection_config
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

    # Resume the run via the checkpoint pipeline surface (the legacy ``training.run_id``
    # key was removed): a RunSource (fork=false) resolves the run's last.ckpt and a
    # WarmStartLoader restores it so Lightning's ckpt_path continues the global step.
    with open_dict(cfg):
        cfg.training.checkpoint = {
            "source": {
                "_target_": "anemoi.training.checkpoint.sources.run.RunSource",
                "run_id": checkpoint_dir.name,
                "fork": False,
            },
            "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"},
        }
    cfg.training.max_epochs = 3
    trainer = AnemoiTrainer(cfg)
    trainer.train()

    expected_global_step = int(cfg.training.max_epochs * cfg.dataloader.limit_batches.training)
    assert (
        trainer.model.trainer.global_step == expected_global_step
    ), f"Expected global_step={expected_global_step}, got {trainer.model.trainer.global_step}"

    assert len(list(checkpoint_dir.glob("anemoi-by_epoch-*.ckpt"))) == 3, "Expected 3 checkpoints after second run"


@skip_if_offline
@pytest.mark.multigpu
@pytest.mark.slow
def test_restart_training_ddp(
    gnn_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
    tmp_path: Path,
) -> None:
    """Warm-start resume under real multi-GPU DDP — the HPC deployment path.

    Mirrors :func:`test_restart_training` but runs across the job's ranks (one task
    per GPU), which the single-process tests never exercise: rank-0/rank-N
    coordination on the resolved checkpoint, optimizer/loop restore across ranks, and
    no deadlock. The output root is a shared, job-scoped directory so every rank
    resolves the same ``last.ckpt`` — pytest's per-process ``tmp_path`` diverges across
    ranks. Launch with ``srun ... python -m pytest ... --multigpu`` on a 2-GPU
    allocation.
    """
    cfg, url = gnn_config
    get_test_archive(url)

    # Every rank must agree on the output location to resolve the same checkpoint on
    # resume. SLURM_JOB_ID is identical across a job's ranks and $SCRATCH is shared;
    # falls back to the per-process tmp_path when run outside SLURM.
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    shared_root = Path(os.environ.get("SCRATCH", str(tmp_path))) / f"ddp_ckpt_resume_{job_id}"
    with open_dict(cfg):
        cfg.system.hardware.accelerator = "cuda"
        cfg.system.hardware.num_gpus_per_node = 2
        cfg.system.output.root = str(shared_root)

    AnemoiTrainer(cfg).train()
    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)
    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert len(run_dirs) == 1, f"Expected exactly one run_id directory, found {[d.name for d in run_dirs]}"
    checkpoint_dir = run_dirs[0]

    # Resume across ranks via the checkpoint pipeline surface (RunSource resolves the
    # same last.ckpt on every rank; WarmStartLoader hands its path to Lightning's
    # ckpt_path so all ranks restore optimizer/loop state and the global step continues).
    with open_dict(cfg):
        cfg.training.checkpoint = {
            "source": {
                "_target_": "anemoi.training.checkpoint.sources.run.RunSource",
                "run_id": checkpoint_dir.name,
                "fork": False,
            },
            "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"},
        }
    cfg.training.max_epochs = 3
    trainer = AnemoiTrainer(cfg)
    trainer.train()

    expected_global_step = int(cfg.training.max_epochs * cfg.dataloader.limit_batches.training)
    assert (
        trainer.model.trainer.global_step == expected_global_step
    ), f"DDP resume: expected global_step={expected_global_step}, got {trainer.model.trainer.global_step}"


@skip_if_offline
@pytest.mark.slow
@pytest.mark.parametrize("config_fixture", ["lam_config", "ensemble_config", "multidatasets_config"])
def test_restart_training_architectures(
    config_fixture: str,
    request: pytest.FixtureRequest,
    get_test_archive: GetTestArchive,
) -> None:
    """Warm-start resume on the architectures models actually use (LAM, ensemble, multi-dataset).

    ``test_restart_training`` covers GNN only; these exercise the non-GNN paths — including
    the multi-dataset metadata handling — through the same resume cycle: train, then
    ``RunSource(fork=false)`` + ``WarmStartLoader``, and assert the global step continues.
    A strict warm-start load that does not round-trip an architecture's state dict fails
    here — this is how the tendency-processor regression was caught.
    """
    cfg, urls = request.getfixturevalue(config_fixture)
    for url in urls if isinstance(urls, list) else [urls]:
        get_test_archive(url)

    # Drop plotting / rollout-eval callbacks: they are not part of the resume contract
    # and pull optional plotting deps (datashader) — same as the checkpoint fixtures.
    with open_dict(cfg):
        cfg.diagnostics.plot.callbacks = []
        cfg.diagnostics.callbacks = []

    AnemoiTrainer(cfg).train()
    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)
    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    assert len(run_dirs) == 1, f"Expected exactly one run_id directory, found {[d.name for d in run_dirs]}"
    checkpoint_dir = run_dirs[0]

    with open_dict(cfg):
        cfg.training.checkpoint = {
            "source": {
                "_target_": "anemoi.training.checkpoint.sources.run.RunSource",
                "run_id": checkpoint_dir.name,
                "fork": False,
            },
            "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"},
        }
    cfg.training.max_epochs = 3
    trainer = AnemoiTrainer(cfg)
    trainer.train()

    expected_global_step = int(cfg.training.max_epochs * cfg.dataloader.limit_batches.training)
    assert (
        trainer.model.trainer.global_step == expected_global_step
    ), f"{config_fixture} resume: expected global_step={expected_global_step}, got {trainer.model.trainer.global_step}"


@skip_if_offline
@pytest.mark.slow
def test_inference_checkpoint_round_trips(
    gnn_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    """A training run's inference checkpoint loads back and carries the inference contract.

    ``AnemoiCheckpoint`` writes an ``inference-*.ckpt`` next to each Lightning checkpoint;
    it is what anemoi-inference consumes downstream, and the existing UUID test never loads
    it back. Train, then assert the saved inference checkpoint ``torch.load``s into a model
    that carries weights and that its metadata satisfies the inference schema — the
    consumption contract — without taking a dependency on anemoi-inference.
    """
    import torch

    from anemoi.utils.checkpoints import load_metadata

    cfg, url = gnn_config
    get_test_archive(url)
    AnemoiTrainer(cfg).train()

    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)
    run_dir = next(item for item in output_dir.iterdir() if item.is_dir())
    inference_ckpts = sorted(run_dir.glob("inference-*.ckpt"))
    assert inference_ckpts, f"No inference checkpoint was saved in {run_dir}"
    inference_ckpt = inference_ckpts[0]

    # Loads back as a usable model carrying weights.
    model = torch.load(inference_ckpt, map_location="cpu", weights_only=False)
    assert sum(p.numel() for p in model.parameters()) > 0, "Inference checkpoint model has no weights"

    # Carries the metadata anemoi-inference consumes (the consumption contract).
    metadata = load_metadata(inference_ckpt)
    assert_keys_exist(metadata, PARTIAL_METADATA_SCHEMA)


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
def test_training_cycle_temporal_downscaler(
    temporal_downscaler_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    """Full training-cycle smoke-test for the temporal downscaler task."""
    cfg, url = temporal_downscaler_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_temporal_downscaler(temporal_downscaler_config: tuple[DictConfig, str]) -> None:
    """Schema-level validation for the temporal downscaler config."""
    cfg, _ = temporal_downscaler_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_edm_transport(edm_transport_config: tuple[DictConfig, str], get_test_archive: callable) -> None:
    cfg, url = edm_transport_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_edm_transport(edm_transport_config: tuple[DictConfig, str]) -> None:
    cfg, _ = edm_transport_config
    BaseSchema(**cfg)


def test_config_validation_stochastic_interpolant(stochastic_interpolant_config: tuple[DictConfig, str]) -> None:
    cfg, _ = stochastic_interpolant_config
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
def test_training_cycle_multidatasets_edm_transport(
    multidatasets_edm_transport_config: tuple[DictConfig, list[str]],
    get_test_archive: callable,
) -> None:
    cfg, urls = multidatasets_edm_transport_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_temporal_downscaler_ensemble(
    temporal_downscaler_ensemble_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = temporal_downscaler_ensemble_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_keys_exist(trainer.metadata, PARTIAL_METADATA_SCHEMA)


def test_config_validation_temporal_downscaler_ensemble(
    temporal_downscaler_ensemble_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = temporal_downscaler_ensemble_config
    BaseSchema(**cfg)


@skip_if_offline
@pytest.mark.slow
def test_evaluator(
    gnn_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    cfg, url = gnn_config
    get_test_archive(url)
    training_cfg = deepcopy(cfg)
    training_cfg.diagnostics.plot.callbacks = []
    training_cfg.dataloader.limit_batches.validation = 0
    AnemoiTrainer(training_cfg).train()

    output_dir = Path(cfg.system.output.root + "/" + cfg.system.output.checkpoints.root)
    assert output_dir.exists(), f"Checkpoint directory not found at: {output_dir}"
    run_dirs = [item for item in output_dir.iterdir() if item.is_dir()]
    checkpoint_dir = run_dirs[0]

    cfg.training.checkpoint = {
        "source": {
            "_target_": "anemoi.training.checkpoint.sources.run.RunSource",
            "run_id": checkpoint_dir.name,
        },
        "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WeightsOnlyLoader"},
    }
    evaluator = AnemoiEvaluator(cfg)
    evaluator.evaluate()
