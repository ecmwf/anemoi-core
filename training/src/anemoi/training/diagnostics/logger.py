# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the gitrms of the Apache Licence Version 2.0
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

from anemoi.training.schemas.base_schema import convert_to_omegaconf

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(
    diagnostics_config: Any,
    run_id: Any,
    fork_run_id: Any,
    paths: Any,
    config: Any,
    **kwargs,
) -> None:
    del kwargs
    if not diagnostics_config.log.mlflow.enabled:
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
    logger_config["save_dir"] = getattr(logger_config, "save_dir", str(paths.logs.mlflow))

    logger = instantiate(
        logger_config,
        run_id=run_id,
        fork_run_id=fork_run_id,
    )

    if logger.log_terminal:
        logger.log_terminal_output(artifact_save_dir=paths.plots)
    if logger.log_system:
        logger.log_system_metrics()

    return logger


def get_wandb_logger(
    diagnostics_config: Any,
    run_id: Any,
    paths: Any,
    model: pl.LightningModule,
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

    try:
        from pytorch_lightning.loggers.wandb import WandbLogger
    except ImportError as err:
        msg = "To activate W&B logging, please install `wandb` as an optional dependency."
        raise ImportError(msg) from err

    logger = WandbLogger(
        project=diagnostics_config.log.wandb.project,
        entity=diagnostics_config.log.wandb.entity,
        id=run_id,
        save_dir=save_dir,
        offline=diagnostics_config.log.wandb.offline,
        log_model=diagnostics_config.log.wandb.log_model,
        resume=run_id is not None,
    )
    if diagnostics_config.log.wandb.gradients or diagnostics_config.log.wandb.parameters:
        if diagnostics_config.log.wandb.gradients and diagnostics_config.log.wandb.parameters:
            log_ = "all"
        elif diagnostics_config.log.wandb.gradients:
            log_ = "gradients"
        else:
            log_ = "parameters"
        logger.watch(model, log=log_, log_freq=diagnostics_config.log.interval, log_graph=False)

    return logger
