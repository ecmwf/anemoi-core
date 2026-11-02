# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema
from anemoi.utils.testing import skip_if_offline

LOGGER = logging.getLogger(__name__)


def test_config_validation_global_config(global_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = global_config
    BaseSchema(**cfg)


def test_config_validation_accepts_unknown_projection_kind(global_config: tuple[DictConfig, str, str]) -> None:
    # projection_kind is a free-form string; unknown values are validated lazily at
    # runtime in Projection.from_kind, not at schema load time.
    cfg, _, _ = global_config
    cfg.diagnostics.plot.settings.projection_kind = "invalid_projection"
    validated = BaseSchema(**cfg)
    assert validated.diagnostics.plot.settings.projection_kind == "invalid_projection"


def test_config_without_validation_accepts_invalid_projection_kind(global_config: tuple[DictConfig, str, str]) -> None:
    cfg, _, _ = global_config
    cfg.config_validation = False
    cfg.diagnostics.plot.settings.projection_kind = "invalid_projection"
    cfg_obj = OmegaConf.to_object(cfg)
    unvalidated = UnvalidatedBaseSchema(**DictConfig(cfg_obj))
    assert unvalidated.diagnostics.plot.settings.projection_kind == "invalid_projection"


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


def test_config_validation_stretched(stretched_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = stretched_config
    BaseSchema(**cfg)


def test_config_validation_multidatasets(multidatasets_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = multidatasets_config
    BaseSchema(**cfg)


def test_config_validation_lam(lam_config: DictConfig) -> None:
    cfg, _ = lam_config
    BaseSchema(**cfg)


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


def test_config_validation_hierarchical(hierarchical_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = hierarchical_config
    BaseSchema(**cfg)


def test_config_validation_autoencoder(autoencoder_config: tuple[DictConfig, list[str]]) -> None:
    cfg, _ = autoencoder_config
    BaseSchema(**cfg)


def test_config_validation_temporal_downscaler(temporal_downscaler_config: tuple[DictConfig, str]) -> None:
    """Schema-level validation for the temporal downscaler config."""
    cfg, _ = temporal_downscaler_config
    BaseSchema(**cfg)


def test_config_validation_edm_transport(edm_transport_config: tuple[DictConfig, str]) -> None:
    cfg, _ = edm_transport_config
    BaseSchema(**cfg)


def test_config_validation_stochastic_interpolant(stochastic_interpolant_config: tuple[DictConfig, str]) -> None:
    cfg, _ = stochastic_interpolant_config
    BaseSchema(**cfg)


def test_config_validation_temporal_downscaler_ensemble(
    temporal_downscaler_ensemble_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = temporal_downscaler_ensemble_config
    BaseSchema(**cfg)


def test_config_validation_offset_forecaster_config(offset_forecaster_config: tuple[DictConfig, str]) -> None:
    cfg, _ = offset_forecaster_config
    BaseSchema(**cfg)


def test_config_validation_offset_forecaster_tendency_transport(
    offset_forecaster_tendency_transport_config: tuple[DictConfig, str],
) -> None:
    cfg, _ = offset_forecaster_tendency_transport_config
    BaseSchema(**cfg)
