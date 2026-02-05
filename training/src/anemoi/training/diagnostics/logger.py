# (C) Copyright 2024 Anemoi contributors.

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

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(
    run_id: str,
    fork_run_id: str,
    paths: DictConfig,
    logger_config: DictConfig,
) -> None:
    mlflow_config = logger_config.mlflow
    if not mlflow_config.enabled:
        LOGGER.debug("MLFlow logging is disabled.")
        return None

    mlflow_config = OmegaConf.to_container(mlflow_config, resolve=True)
    EXTRA_KEYS = ['enabled']
    for key in EXTRA_KEYS:
        mlflow_config.pop(key, None)

    # backward compatibility to not break configs
    mlflow_config["_target_"] = mlflow_config.get(
        "_target_",
        "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger",
    )
    mlflow_config["save_dir"] = mlflow_config.get("save_dir", str(paths.logs.mlflow))
    logger = instantiate(
        mlflow_config,
        run_id=run_id,
        fork_run_id=fork_run_id,
    )
    if logger.log_terminal:
        logger.log_terminal_output(artifact_save_dir=paths.plots)
    if logger.log_system:
        logger.log_system_metrics()

    return logger


def get_wandb_logger(
    run_id: str,
    paths: DictConfig,
    model: pl.LightningModule,
    logger_config: DictConfig,
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

    save_dir = paths.logs.wandb
    wandb_config = logger_config.wandb
    gradients = wandb_config.gradients
    parameters = wandb_config.parameters

    # backward compatibility to not break configs
    interval = getattr(wandb_config, "interval", 100)

    if not wandb_config.enabled:
        LOGGER.debug("Weights & Biases logging is disabled.")
        return None

    wandb_config = OmegaConf.to_container(wandb_config, resolve=True)
    EXTRA_KEYS = ["gradients", "parameters", "interval",'enabled']
    for key in EXTRA_KEYS:
        wandb_config.pop(key, None)

    try:
        logger = instantiate(
            wandb_config,
            id=run_id,
            save_dir=save_dir,
            resume=run_id is not None,
        )
    except ImportError as err:
        msg = "To activate W&B logging, please install `wandb` as an optional dependency."
        raise ImportError(msg) from err

    if gradients or parameters:
        if gradients and parameters:
            log_ = "all"
        elif gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=interval, log_graph=False)

    return logger
