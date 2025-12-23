# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(
    diagnostics_config: Any,
    run_id: Any,
    fork_run_id: Any,
    paths: Any,
    config: Any,
    **kwargs,
) -> None:
    if not diagnostics_config.log.mlflow.enabled:
        LOGGER.debug("MLFlow logging is disabled.")
        return None

    logger_config = OmegaConf.to_container(diagnostics_config.log.mlflow)
    del logger_config["enabled"]

    # backward compatibility to not break configs
    logger_config["_target_"] = logger_config.get(
        "_target_",
        "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger",
    )
    logger_config["save_dir"] = logger_config.get("save_dir", str(paths.output.logs.mlflow))

    logger = instantiate(
        logger_config,
        run_id=config.training.run_id,
        fork_run_id=config.training.fork_run_id,
    )

    if logger.log_terminal:
        logger.log_terminal_output(artifact_save_dir=paths.output.get("plots"))
    if logger.log_system:
        logger.log_system_metrics()

    return logger


def get_tensorboard_logger(diagnostics_config: Any, paths: Any, **kwargs) -> pl.loggers.TensorBoardLogger | None:
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
    if not diagnostics_config.log.tensorboard.enabled:
        LOGGER.debug("Tensorboard logging is disabled.")
        return None

    from pytorch_lightning.loggers import TensorBoardLogger

    return TensorBoardLogger(
        save_dir=paths.output.logs.tensorboard,
        log_graph=False,
    )


def get_wandb_logger(
    diagnostics_config: Any,
    run_id: Any,
    paths: Any,
    model: pl.LightningModule,
    config: Any,
    **kwargs,
) -> pl.loggers.WandbLogger | None:
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
    if not diagnostics_config.log.wandb.enabled:
        LOGGER.debug("Weights & Biases logging is disabled.")
        return None

    try:
        from pytorch_lightning.loggers.wandb import WandbLogger
    except ImportError as err:
        msg = "To activate W&B logging, please install `wandb` as an optional dependency."
        raise ImportError(msg) from err

    logger = WandbLogger(
        project=diagnostics_config.log.wandb.project,
        entity=diagnostics_config.log.wandb.entity,
        id=run_id,
        save_dir=paths.output.logs.wandb,
        offline=diagnostics_config.log.wandb.offline,
        log_model=diagnostics_config.log.wandb.log_model,
        resume=run_id is not None,
    )
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    if diagnostics_config.log.wandb.gradients or diagnostics_config.log.wandb.parameters:
        if diagnostics_config.log.wandb.gradients and diagnostics_config.log.wandb.parameters:
            log_ = "all"
        elif diagnostics_config.log.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=diagnostics_config.log.interval, log_graph=False)

    return logger
