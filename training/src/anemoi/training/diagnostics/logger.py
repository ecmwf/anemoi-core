# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping

import pytorch_lightning as pl

from anemoi.training.builders.loggers import build_mlflow_logger_from_config
from anemoi.training.builders.loggers import build_tensorboard_logger_from_config
from anemoi.training.builders.loggers import build_wandb_logger_from_config
from anemoi.training.config_types import ConfigBase

LOGGER = logging.getLogger(__name__)


def get_mlflow_logger(config: ConfigBase | Mapping) -> pl.loggers.MLFlowLogger | None:
    logger = build_mlflow_logger_from_config(config)
    if logger is None:
        LOGGER.debug("MLFlow logging is disabled.")
    return logger


def get_tensorboard_logger(config: ConfigBase | Mapping) -> pl.loggers.TensorBoardLogger | None:
    """Setup TensorBoard experiment logger.

    Parameters
    ----------
    config : ConfigBase | Mapping
        Job configuration

    Returns
    -------
    pl.loggers.TensorBoardLogger | None
        Logger object, or None

    """
    logger = build_tensorboard_logger_from_config(config)
    if logger is None:
        LOGGER.debug("Tensorboard logging is disabled.")
    return logger


def get_wandb_logger(config: ConfigBase | Mapping, model: pl.LightningModule) -> pl.loggers.WandbLogger | None:
    """Setup Weights & Biases experiment logger.

    Parameters
    ----------
    config : ConfigBase | Mapping
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
    logger = build_wandb_logger_from_config(config, model)
    if logger is None:
        LOGGER.debug("Weights & Biases logging is disabled.")
    return logger
