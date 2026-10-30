# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable

import pytest
from omegaconf import DictConfig
from omegaconf import open_dict

from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_global(
    global_config: tuple[DictConfig, str, str],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, url, _ = global_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


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
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, urls = stretched_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_multidatasets(
    multidatasets_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, urls = multidatasets_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_lam(
    lam_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, urls = lam_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


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


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_ensemble(
    ensemble_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, url = ensemble_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


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


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_autoencoder(
    autoencoder_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, urls = autoencoder_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_temporal_downscaler(
    temporal_downscaler_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    """Full training-cycle smoke-test for the temporal downscaler task."""
    cfg, url = temporal_downscaler_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_edm_transport(
    edm_transport_config: tuple[DictConfig, str],
    get_test_archive: callable,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, url = edm_transport_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
@pytest.mark.mlflow
def test_training_cycle_mlflow_dry_run(
    mlflow_dry_run_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
) -> None:
    from anemoi.training.commands.mlflow import prepare_mlflow_run_id

    cfg, url = mlflow_dry_run_config

    # Generate a dry run ID and attach the run to it. `training.run_id` was removed;
    # a prepared run is now named by a resume RunIdSource. The prepared run has no
    # checkpoint directory yet, so this also exercises the dry-run gate: the trainer
    # must clear `start_from_checkpoint` and start fresh rather than looking for a
    # checkpoint that was never written.
    run_id, _ = prepare_mlflow_run_id(
        config=cfg,
    )
    with open_dict(cfg):
        cfg.training.checkpoint = {
            "source": {
                "_target_": "anemoi.training.checkpoint.sources.run.RunIdSource",
                "run_id": run_id,
                "fork": False,
            },
        }

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
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, urls = multidatasets_edm_transport_config
    for url in urls:
        get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_temporal_downscaler_ensemble(
    temporal_downscaler_ensemble_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, url = temporal_downscaler_ensemble_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_offset_forecaster(
    offset_forecaster_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    cfg, url = offset_forecaster_config
    get_test_archive(url)
    trainer = AnemoiTrainer(cfg)
    trainer.train()
    assert_metadata_schema(trainer.metadata)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_offset_forecaster_tendency_transport(
    offset_forecaster_tendency_transport_config: tuple[DictConfig, str],
    get_test_archive: GetTestArchive,
    assert_metadata_schema: Callable[[dict], None],
) -> None:
    """Train the tendency transport path with irregular inputs and two forecast lead times."""
    cfg, url = offset_forecaster_tendency_transport_config
    get_test_archive(url)

    trainer = AnemoiTrainer(cfg)
    trainer.train()

    assert trainer.task.name == "offset-forecaster"
    assert trainer.datamodule.statistics_tendencies["data"]["lead_times"] == ["6h", "12h"]
    assert trainer.model.model.pre_processors_tendencies["data"].lead_times == ["6h", "12h"]
    assert trainer.metadata["metadata_inference"]["data"]["timesteps"]["input_offsets"] == ["-12h", "0h"]
    assert trainer.metadata["metadata_inference"]["data"]["timesteps"]["output_offsets"] == ["6h", "12h"]
    assert_metadata_schema(trainer.metadata)
