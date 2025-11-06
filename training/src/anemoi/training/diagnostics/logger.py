# (C) Copyright 2024 Anemoi contributors.
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

import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(config: BaseSchema) -> None:
    if not config.diagnostics.log.mlflow.enabled:
        LOGGER.debug("MLFlow logging is disabled.")
        return None

    # 35 retries allow for 1 hour of server downtime
    http_max_retries = config.diagnostics.log.mlflow.http_max_retries

    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(http_max_retries)
    os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = str(http_max_retries + 1)
    # these are the default values, but set them explicitly in case they change
    os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = "2"
    os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_JITTER"] = "1"

    from anemoi.training.diagnostics.mlflow import LOG_MODEL
    from anemoi.training.diagnostics.mlflow import MAX_PARAMS_LENGTH
    from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger

    resumed = config.training.run_id is not None
    forked = config.training.fork_run_id is not None

    save_dir = config.hardware.paths.logs.mlflow

    offline = config.diagnostics.log.mlflow.offline
    if not offline:
        tracking_uri = config.diagnostics.log.mlflow.tracking_uri
        LOGGER.info("AnemoiMLFlow logging to %s", tracking_uri)
    else:
        tracking_uri = None

    if (resumed or forked) and (offline):  # when resuming or forking offline -
        # tracking_uri = ${hardware.paths.logs.mlflow}
        tracking_uri = str(save_dir)
    # create directory if it does not exist
    Path(config.hardware.paths.logs.mlflow).mkdir(parents=True, exist_ok=True)

    log_hyperparams = True
    if resumed and not config.diagnostics.log.mlflow.on_resume_create_child:
        LOGGER.info(
            (
                "Resuming run without creating child run - MLFlow logs will not update the"
                "initial runs hyperparameters with those of the resumed run."
                "To update the initial run's hyperparameters, set "
                "`diagnostics.log.mlflow.on_resume_create_child: True`."
            ),
        )
        log_hyperparams = False

    max_params_length = getattr(config.diagnostics.log.mlflow, "max_params_length", MAX_PARAMS_LENGTH)
    LOGGER.info("Maximum number of params allowed to be logged is: %s", max_params_length)
    log_model = getattr(config.diagnostics.log.mlflow, "log_model", LOG_MODEL)

    logger = AnemoiMLflowLogger(
        experiment_name=config.diagnostics.log.mlflow.experiment_name,
        project_name=config.diagnostics.log.mlflow.project_name,
        tracking_uri=tracking_uri,
        save_dir=save_dir,
        run_name=config.diagnostics.log.mlflow.run_name,
        run_id=config.training.run_id,
        fork_run_id=config.training.fork_run_id,
        log_model=log_model,
        offline=offline,
        resumed=resumed,
        forked=forked,
        log_hyperparams=log_hyperparams,
        authentication=config.diagnostics.log.mlflow.authentication,
        on_resume_create_child=config.diagnostics.log.mlflow.on_resume_create_child,
        max_params_length=max_params_length,
    )

    if config.diagnostics.log.mlflow.terminal:
        logger.log_terminal_output(artifact_save_dir=config.hardware.paths.plots)
    if config.diagnostics.log.mlflow.system:
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
