# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from pytorch_lightning.callbacks import LearningRateMonitor as pl_LearningRateMonitor
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging as pl_StochasticWeightAveraging

LOGGER = logging.getLogger(__name__)


class LearningRateMonitor(pl_LearningRateMonitor):
    """Provide LearningRateMonitor from pytorch_lightning as a callback."""

    def __init__(
        self,
        logging_interval: str = "step",
        log_momentum: bool = False,
    ) -> None:
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum)


class StochasticWeightAveraging(pl_StochasticWeightAveraging):
    """Provide StochasticWeightAveraging from pytorch_lightning as a callback."""

    def __init__(
        self,
        swa_lrs: float,
        max_epochs: int,
        swa_epoch_start: int | None = None,
        annealing_epochs: int | None = None,
        annealing_strategy: str | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Stochastic Weight Averaging Callback.

        Parameters
        ----------
        swa_lrs : float
            Stochastic Weight Averaging Learning Rate
        max_epochs : int
            Total number of training epochs, used to compute default epoch start and annealing epochs
        swa_epoch_start : int, optional
            Epoch to start SWA, by default 0.75 * max_epochs
        annealing_epochs : int, optional
            Number of annealing epochs, by default 0.25 * max_epochs
        annealing_strategy : str, optional
            Annealing strategy, by default 'cos'
        device : str, optional
            Device to use, by default None
        """
        kwargs["swa_lrs"] = swa_lrs
        kwargs["swa_epoch_start"] = (
            swa_epoch_start
            if swa_epoch_start is not None
            else min(
                int(0.75 * max_epochs),
                max_epochs - 1,
            )
        )
        kwargs["annealing_epochs"] = (
            annealing_epochs if annealing_epochs is not None else max(int(0.25 * max_epochs), 1)
        )
        kwargs["annealing_strategy"] = annealing_strategy or "cos"
        kwargs["device"] = device

        super().__init__(**kwargs)
