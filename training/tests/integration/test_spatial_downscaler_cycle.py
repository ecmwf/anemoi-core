# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""End-to-end training smokes for the spatial residual downscaler.

Composes the ``downscaler.yaml`` template (era5 -> cerra residual downscaling), builds a
real nearest-neighbour interpolation matrix from the two dataset grids at fixture time,
supplies explicit residual statistics, and runs ``AnemoiTrainer(cfg).train()`` for one
optimizer step under both transport objectives (EDM diffusion and stochastic
interpolant). ``compute_residual`` is wrapped with a call counter to prove the residual
transformation is reached from the production path, not assumed.
"""

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.layers.residual import InterpolationConnection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportResidualModelEncProcDec
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer
from anemoi.utils.testing import GetTestArchive
from anemoi.utils.testing import skip_if_offline

os.environ["ANEMOI_BASE_SEED"] = "42"  # need to set base seed if running on github runners

LOGGER = logging.getLogger(__name__)

# Same crop as training/tests/integration/config/test_spatial_downscaler.yaml (N, W, S, E).
CERRA_AREA = (55.0, 0.0, 50.0, 8.0)


@pytest.fixture
def spatial_downscaler_config(
    testing_modifications_with_temp_dir: DictConfig,
    get_tmp_path,
) -> tuple[DictConfig, list[str]]:
    """Compose a runnable configuration for the spatial residual downscaler."""
    with initialize(
        version_base=None,
        config_path="../../src/anemoi/training/config",
        job_name="test_spatial_downscaler",
    ):
        template = compose(config_name="downscaler")

    use_case_modifications = OmegaConf.load(
        Path.cwd() / "training/tests/integration/config/test_spatial_downscaler.yaml",
    )
    assert isinstance(use_case_modifications, DictConfig)

    tmp_dir_dataset, url_dataset = get_tmp_path(use_case_modifications.system.input.dataset)
    tmp_dir_dataset_b, url_dataset_b = get_tmp_path(use_case_modifications.system.input.dataset_b)
    use_case_modifications.system.input.dataset = str(tmp_dir_dataset)
    use_case_modifications.system.input.dataset_b = str(tmp_dir_dataset_b)

    # The use-case modifications add an `area` crop to the cerra dataset_config and fill
    # the (empty-by-default) residual_prediction / direct_prediction mappings.
    OmegaConf.set_struct(template.dataloader, False)
    OmegaConf.set_struct(template.model, False)
    cfg = OmegaConf.merge(template, testing_modifications_with_temp_dir, use_case_modifications)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)

    OmegaConf.set_struct(cfg, False)

    # The residual linearity guard only allows affine pre-processors on datasets feeding a
    # residual pair; drop the (non-affine) constant imputer that data:multi puts on cerra.
    cfg.data.datasets.cerra.processors.pop("const_imputer", None)

    # Only the residual target is predicted, so only it can contribute to the loss.
    cfg.training.training_loss.datasets = OmegaConf.create(
        {"cerra": cfg.training.training_loss.datasets.cerra},
    )

    return cfg, [url_dataset, url_dataset_b]


def _lat_lon_to_xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitudes, dtype=np.float64))
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


def _attach_residual_inputs(cfg: DictConfig, tmp_path: Path) -> None:
    """Build the interpolation matrix and residual statistics from the downloaded datasets.

    Must run after the test archives have been extracted. Writes a nearest-neighbour
    (target grid x source grid) scipy sparse matrix and a JSON residual-statistics file,
    then points the configuration at both.
    """
    from anemoi.datasets import open_dataset

    source = open_dataset(dataset=cfg.system.input.dataset)
    target = open_dataset(dataset=cfg.system.input.dataset_b, area=CERRA_AREA)

    source_xyz = _lat_lon_to_xyz(source.latitudes, source.longitudes)
    target_xyz = _lat_lon_to_xyz(target.latitudes, target.longitudes)

    from scipy.spatial import cKDTree

    _, nearest = cKDTree(source_xyz).query(target_xyz, k=1)
    n_target, n_source = target_xyz.shape[0], source_xyz.shape[0]
    matrix = scipy.sparse.csr_matrix(
        (np.ones(n_target, dtype=np.float32), (np.arange(n_target), nearest)),
        shape=(n_target, n_source),
    )
    interpolation_path = tmp_path / "era5_to_cerra_nearest.npz"
    scipy.sparse.save_npz(interpolation_path, matrix)

    n_variables = len(np.asarray(target.statistics["stdev"]))
    statistics_path = tmp_path / "cerra_residual_statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "mean": [0.0] * n_variables,
                "stdev": [1.0] * n_variables,
                "minimum": [-1.0] * n_variables,
                "maximum": [1.0] * n_variables,
            },
        ),
    )

    cfg.model.residual = OmegaConf.create(
        {
            "_target_": "anemoi.models.layers.residual.InterpolationConnection",
            "interpolation_file_path": str(interpolation_path),
        },
    )
    cfg.system.input.statistics_residuals = {"cerra": str(statistics_path)}


def _switch_to_stochastic_interpolant(cfg: DictConfig) -> None:
    cfg.training.transport.objective = "stochastic_interpolant"
    cfg.model.model.transport.objective = "stochastic_interpolant"
    cfg.model.model.transport.training_condition = OmegaConf.create({"distribution": "uniform_time"})


def _install_compute_residual_counter(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls = {"count": 0}
    original = AnemoiTransportResidualModelEncProcDec.compute_residual

    def counting_compute_residual(self, *args, **kwargs):
        calls["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AnemoiTransportResidualModelEncProcDec, "compute_residual", counting_compute_residual)
    return calls


def _assert_downscaler_training_outcome(trainer: AnemoiTrainer, compute_residual_calls: dict) -> None:
    module = trainer.model
    model = module.model.model
    assert isinstance(model, AnemoiTransportResidualModelEncProcDec)
    assert isinstance(model.residual["cerra"], InterpolationConnection)

    # (a) The residual training mode persisted the same-offset reference alignment.
    reference_positions = model._output_reference_positions.tolist()
    assert -1 not in reference_positions, f"reference positions never set: {reference_positions}"
    assert reference_positions == [0]  # input_offsets [0] / output_offsets [0]

    # (b) The task recorded its integer dataset-relative offsets in the metadata.
    metadata = trainer.metadata
    assert metadata["input_datasets"] == ["era5", "cerra"]
    assert metadata["output_datasets"] == ["cerra"]
    for dataset_name in ("era5", "cerra"):
        timesteps = metadata["metadata_inference"][dataset_name]["timesteps"]
        assert timesteps["input_offsets"] == [0]
        assert timesteps["output_offsets"] == [0]

    # (c) At least one optimizer step ran and the training loss is finite.
    assert module.trainer.global_step >= 1
    train_loss_keys = [key for key in module.trainer.callback_metrics if key.startswith("train_") and "loss" in key]
    assert train_loss_keys, f"no training loss was logged: {list(module.trainer.callback_metrics)}"
    for key in train_loss_keys:
        value = float(module.trainer.callback_metrics[key])
        assert math.isfinite(value), f"{key} is not finite: {value}"

    # The residual transformation was actually reached from the production training path.
    assert compute_residual_calls["count"] >= 1, "compute_residual was never called during training"


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_spatial_downscaler_edm(
    spatial_downscaler_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, urls = spatial_downscaler_config
    for url in urls:
        get_test_archive(url)
    _attach_residual_inputs(cfg, tmp_path)

    compute_residual_calls = _install_compute_residual_counter(monkeypatch)

    trainer = AnemoiTrainer(cfg)
    trainer.train()

    _assert_downscaler_training_outcome(trainer, compute_residual_calls)


@skip_if_offline
@pytest.mark.slow
def test_training_cycle_spatial_downscaler_stochastic_interpolant(
    spatial_downscaler_config: tuple[DictConfig, list[str]],
    get_test_archive: GetTestArchive,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, urls = spatial_downscaler_config
    _switch_to_stochastic_interpolant(cfg)
    for url in urls:
        get_test_archive(url)
    _attach_residual_inputs(cfg, tmp_path)

    compute_residual_calls = _install_compute_residual_counter(monkeypatch)

    trainer = AnemoiTrainer(cfg)
    trainer.train()

    _assert_downscaler_training_outcome(trainer, compute_residual_calls)


def test_config_validation_spatial_downscaler(
    spatial_downscaler_config: tuple[DictConfig, list[str]],
    tmp_path: Path,
) -> None:
    """Schema-level validation of the composed downscaler config (no dataset download).

    The residual/statistics paths only need to exist as values, not on disk, for schema
    validation, so placeholders are used instead of the downloaded-dataset fixtures.
    """
    cfg, _ = spatial_downscaler_config
    cfg.model.residual = OmegaConf.create(
        {
            "_target_": "anemoi.models.layers.residual.InterpolationConnection",
            "interpolation_file_path": str(tmp_path / "placeholder.npz"),
        },
    )
    cfg.system.input.statistics_residuals = {"cerra": str(tmp_path / "placeholder.json")}
    BaseSchema(**cfg)
