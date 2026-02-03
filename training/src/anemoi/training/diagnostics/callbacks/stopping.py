# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import time
from datetime import timedelta
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only

from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class TimeLimit(pl.callbacks.Callback):
    """Callback to stop the training process after a given time limit."""

    def __init__(self, config: DictConfig, limit: int | str, record_file: str | None = None) -> None:
        """Initialise the TimeLimit callback.

        Parameters
        ----------
        limit : int or str
            The frequency to convert. If an integer, it is assumed to be in hours. If a string, it can be in the format:

            - "1h" for 1 hour
            - "1d" for 1 day
            - "1m" for 1 minute
            - "1s" for 1 second
            - "1:30" for 1 hour and 30 minutes
            - "1:30:10" for 1 hour, 30 minutes and 10 seconds
            - "PT10M" for 10 minutes (ISO8601)

        record_file : str or None
            The file to record the last checkpoint to. If None, no file is written.

        """
        super().__init__()
        self.config = config

        self.limit = frequency_to_timedelta(limit)
        self._record_file = Path(record_file) if record_file is not None else None

        if self._record_file is not None and self._record_file.exists():
            assert self._record_file.is_file(), "The record file must be a file."

        self._start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        _ = pl_module
        self._run_stopping_check(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        _ = pl_module
        self._run_stopping_check(trainer)

    @rank_zero_only
    def _run_stopping_check(self, trainer: pl.Trainer) -> None:
        """Check if the time limit has been reached and stop the training if so.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        """
        if timedelta(seconds=time.time() - self._start_time) < self.limit:
            return

        LOGGER.info("Time limit of %s reached. Stopping training.", frequency_to_string(self.limit))
        trainer.should_stop = True
        self._log_to_file(trainer)

    @rank_zero_only
    def _log_to_file(self, trainer: pl.Trainer) -> None:
        """Log the last checkpoint path to a file if given.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        """
        if self._record_file is not None:
            last_checkpoint = trainer.checkpoint_callback.last_model_path
            self._record_file.parent.mkdir(parents=True, exist_ok=True)

            if self._record_file.exists():
                self._record_file.unlink()

            Path(self._record_file).write_text(str(last_checkpoint))


class EarlyStopping(pl.callbacks.EarlyStopping):
    """Thin wrapper around Pytorch Lightning's EarlyStopping callback."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        """Early stopping callback.

        Set `monitor` to metric to check.
        Common names within `Anemoi` are:
            - `val_{loss_name}_epoch`
            - `train_{loss_name}_epoch`
            - `val_{metric_name}/{param}_{level}/{rollout}` i.e. `val_wmse/v_850/1`
            - `val_{metric_name}/sfc_{param}/{rollout}` i.e. `val_wmse/sfc_2t/1`

        See Pytorch Lightning documentation for more information.
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
        """
        super().__init__(**kwargs)
        self.config = config
