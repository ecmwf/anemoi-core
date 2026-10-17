# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Union

import torch
from pytorch_lightning.callbacks import WeightAveraging as _PLWeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn


class WeightAveraging(_PLWeightAveraging):
    """Base class that averages parameters and synchronises fixed buffers."""

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, use_buffers=use_buffers, **kwargs)


class EMAWeightAveraging(WeightAveraging):
    """Exponential Moving Average weight averaging."""

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = False,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            use_buffers=use_buffers,
            **kwargs,
            avg_fn=get_ema_avg_fn(decay=decay),
        )
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch


class SWAWeightAveraging(WeightAveraging):
    """Stochastic Weight Averaging (running mean).

    Uses the default running-mean function from PyTorch's ``AveragedModel``.
    """

    def __init__(
        self,
        device: Union[torch.device, str, int] | None = None,
        use_buffers: bool = False,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, use_buffers=use_buffers, **kwargs)
        self.update_every_n_steps = update_every_n_steps
        self.update_starting_at_step = update_starting_at_step
        self.update_starting_at_epoch = update_starting_at_epoch
