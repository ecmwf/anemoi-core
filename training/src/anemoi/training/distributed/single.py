# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from pytorch_lightning.strategies import SingleDeviceStrategy as SingleDeviceStrategyLightning

DROPPED_EXTRA_KEYS = [
    "static_graph",
    "use_local_synchronization",
    "broadcast_buffers",
]


class SingleDeviceStrategy(SingleDeviceStrategyLightning):
    """Single device strategy, supporting removing kwargs commonly used in Anemoi distributed strategies."""

    @property
    def read_group_size(self) -> int:
        """Mimics to ensure compatibility with distributed strategies."""
        return 1

    def __init__(
        self,
        device: str = "auto",
        num_gpus_per_model: int = 1,
        num_gpus_per_ensemble: int = 1,
        read_group_size: int = 1,
        **kwargs: dict,
    ) -> None:
        """Initialise the distributed strategy.

        Parameters
        ----------
        device : str
            Device to use for training. Can be "auto", "cpu", or "cuda:<index>".
        num_gpus_per_model : int
            Number of GPUs per model to shard over.
        num_gpus_per_ensemble : int
            Number of GPUs per ensemble.
        read_group_size : int
            Number of GPUs per reader group.
        **kwargs : dict
            Additional keyword arguments.
        """
        for key in DROPPED_EXTRA_KEYS:
            kwargs.pop(key, None)  # Remove any dropped extra keys

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        super().__init__(device=device, **kwargs)

        assert num_gpus_per_model == 1, "SingleDeviceStrategy only supports num_gpus_per_model=1"
        assert num_gpus_per_ensemble == 1, "SingleDeviceStrategy only supports num_gpus_per_ensemble=1"
        assert read_group_size == 1, "SingleDeviceStrategy only supports read_group_size=1"
