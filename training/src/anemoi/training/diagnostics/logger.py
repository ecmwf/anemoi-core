# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(config: BaseSchema) -> None:
    if not config.diagnostics.log.mlflow.enabled:
        LOGGER.debug("MLFlow logging is disabled.")
        return None

    logger_config = OmegaConf.to_container(convert_to_omegaconf(config).diagnostics.log.mlflow)
    del logger_config["enabled"]

    # backward compatibility to not break configs
    logger_config["_target_"] = getattr(
        logger_config,
        "_target",
        "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger",
    )
    logger_config["save_dir"] = getattr(logger_config, "save_dir", str(config.hardware.paths.logs.mlflow))

    logger = instantiate(
        logger_config,
        run_id=config.training.run_id,
        fork_run_id=config.training.fork_run_id,
    )

    if logger.log_terminal:
        logger.log_terminal_output(artifact_save_dir=config.hardware.paths.plots)
    if logger.log_system:
        logger.log_system_metrics()

    return logger


def get_tensorboard_logger(config: DictConfig) -> pl.loggers.TensorBoardLogger | None:
    """Setup TensorBoard experiment logger.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    pl.loggers.TensorBoardLogger | None
        Logger object, or None

    """
    if not config.diagnostics.log.tensorboard.enabled:
        LOGGER.debug("Tensorboard logging is disabled.")
        return None

    from pytorch_lightning.loggers import TensorBoardLogger

    return TensorBoardLogger(
        save_dir=config.hardware.paths.logs.tensorboard,
        log_graph=False,
    )


def get_wandb_logger(config: DictConfig, model: pl.LightningModule) -> pl.loggers.WandbLogger | None:
    """Setup Weights & Biases experiment logger.

    Parameters
    ----------
    config : DictConfig
        Job configuration
    model: GraphForecaster
        Model to watch

    Returns
    -------
    pl.loggers.WandbLogger | None
        Logger object

    Raises
    ------
    ImportError
        If `wandb` is not installed

    """
    if not config.diagnostics.log.wandb.enabled:
        LOGGER.debug("Weights & Biases logging is disabled.")
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
        save_dir=config.hardware.paths.logs.wandb,
        offline=config.diagnostics.log.wandb.offline,
        log_model=config.diagnostics.log.wandb.log_model,
        resume=config.training.run_id is not None,
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    if config.diagnostics.log.wandb.gradients or config.diagnostics.log.wandb.parameters:
        if config.diagnostics.log.wandb.gradients and config.diagnostics.log.wandb.parameters:
            log_ = "all"
        elif config.diagnostics.log.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=config.diagnostics.log.interval, log_graph=False)

    return logger
