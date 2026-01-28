# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.builders.components import build_component
from anemoi.training.config_types import ConfigBase
from anemoi.training.config_types import to_container

if TYPE_CHECKING:

    import pytorch_lightning as pl


def _to_container(value: Any) -> Any:
    if isinstance(value, ConfigBase):
        return to_container(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True)
    return to_container(value) if isinstance(value, dict) else value


def build_mlflow_logger_from_config(
    config: ConfigBase,
) -> pl.loggers.MLFlowLogger | None:
    if not config.diagnostics.log.mlflow.enabled:
        return None

    logger_config = _to_container(config.diagnostics.log.mlflow)
    if isinstance(logger_config, dict):
        logger_config = dict(logger_config)
        logger_config.pop("enabled", None)

    logger = build_component(
        logger_config,
        run_id=config.training.run_id,
        fork_run_id=config.training.fork_run_id,
    )
    if logger.log_terminal:
        logger.log_terminal_output(artifact_save_dir=config.system.output.plots)
    if logger.log_system:
        logger.log_system_metrics()
    return logger


def build_tensorboard_logger_from_config(
    config: ConfigBase,
) -> pl.loggers.TensorBoardLogger | None:
    if not config.diagnostics.log.tensorboard.enabled:
        return None

    from pytorch_lightning.loggers import TensorBoardLogger

    return TensorBoardLogger(
        save_dir=config.system.output.logs.tensorboard,
        log_graph=False,
    )


def build_wandb_logger_from_config(
    config: ConfigBase,
    model: pl.LightningModule | None = None,
) -> pl.loggers.WandbLogger | None:
    if not config.diagnostics.log.wandb.enabled:
        return None

    try:
        from pytorch_lightning.loggers.wandb import WandbLogger
    except ImportError as err:
        msg = "To activate W&B logging, please install `wandb` as an optional dependency."
        raise ImportError(msg) from err

    logger = WandbLogger(
        project=config.diagnostics.log.wandb.project,
        entity=config.diagnostics.log.wandb.entity,
        id=config.training.run_id,
        save_dir=config.system.output.logs.wandb,
        offline=config.diagnostics.log.wandb.offline,
        log_model=config.diagnostics.log.wandb.log_model,
        resume=config.training.run_id is not None,
    )
    logger.log_hyperparams(_to_container(config))
    if config.diagnostics.log.wandb.gradients or config.diagnostics.log.wandb.parameters:
        if model is None:
            msg = "W&B logger requires a model to watch gradients or parameters."
            raise ValueError(msg)
        if config.diagnostics.log.wandb.gradients and config.diagnostics.log.wandb.parameters:
            log_ = "all"
        elif config.diagnostics.log.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=config.diagnostics.log.interval, log_graph=False)
    return logger


__all__ = [
    "build_mlflow_logger_from_config",
    "build_tensorboard_logger_from_config",
    "build_wandb_logger_from_config",
]
