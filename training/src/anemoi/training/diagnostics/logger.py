# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the gitrms of the Apache Licence Version 2.0
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

from anemoi.training.schemas.base_schema import convert_to_omegaconf

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(
    diagnostics_config: DictConfig,
    run_id: str,
    fork_run_id: str,
    paths: DictConfig,
    logger_config: DictConfig,
    **kwargs,
) -> None:
    del kwargs
    if not diagnostics_config.log.mlflow.enabled:
        LOGGER.debug("MLFlow logging is disabled.")
        return None

    logger_config = OmegaConf.to_container(convert_to_omegaconf(logger_config.mlflow))
    del logger_config["enabled"]

    # backward compatibility to not break configs
    logger_config["_target_"] = logger_config.get(
        "_target_",
        "anemoi.training.diagnostics.mlflow.logger.AnemoiMLflowLogger",
    )
    logger_config["save_dir"] = logger_config.get("save_dir", str(paths.logs.mlflow))

    logger = instantiate(
        logger_config,
        run_id=run_id,
        ork_run_id=fork_run_id,
    )
    if logger.log_terminal:
        logger.log_terminal_output(artifact_save_dir=paths.plots)
    if logger.log_system:
        logger.log_system_metrics()

    return logger


def get_wandb_logger(
    diagnostics_config: DictConfig,
    run_id: str,
    paths: DictConfig,
    model: pl.LightningModule,
    logger_config: DictConfig,
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
    del kwargs

    if not diagnostics_config.log.wandb.enabled:
        LOGGER.debug("Weights & Biases logging is disabled.")
        return None

    save_dir = paths.logs.wandb
    logger_config = OmegaConf.to_container(convert_to_omegaconf(logger_config.wandb))

    try:
        logger = instantiate(
            logger_config,
            id=run_id,
            save_dir=save_dir,
            resume=run_id is not None,
        )
    except ImportError as err:
        msg = "To activate W&B logging, please install `wandb` as an optional dependency."
        raise ImportError(msg) from err

    if diagnostics_config.log.wandb.gradients or diagnostics_config.log.wandb.parameters:
        if diagnostics_config.log.wandb.gradients and diagnostics_config.log.wandb.parameters:
            log_ = "all"
        elif diagnostics_config.log.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=diagnostics_config.log.interval, log_graph=False)

    return logger
