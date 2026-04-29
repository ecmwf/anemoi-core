# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor as pl_LearningRateMonitor

LOGGER = logging.getLogger(__name__)


class LearningRateMonitor(pl_LearningRateMonitor):
    """Provide LearningRateMonitor from pytorch_lightning as a callback."""

    def __init__(
        self,
        config: DictConfig,
        logging_interval: str = "step",
        log_momentum: bool = False,
    ) -> None:
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum)
        self.config = config
