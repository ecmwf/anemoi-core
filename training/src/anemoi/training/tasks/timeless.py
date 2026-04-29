# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import torch

from anemoi.training.diagnostics.callbacks.plot_adapter import AutoencoderPlotAdapter
from anemoi.training.tasks.base import BaseSingleStepTask


class BaseTimelessTask(BaseSingleStepTask):
    """Base class for timeless tasks.

    Both input and output are a single snapshot at t=0.
    """

    def __init__(self, **_kwargs) -> None:
        super().__init__(input_offsets=[datetime.timedelta(0)], output_offsets=[datetime.timedelta(0)])

        self._plot_adapter = AutoencoderPlotAdapter(self)

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return "0H"


class Autoencoder(BaseTimelessTask):
    """Autoencoding task implementation."""

    name: str = "autoencoder"


class SpatialDownscaler(BaseTimelessTask):
    """Spatial downscaling task.

    Both input (in_lres, in_hres) and output (out_hres) are single snapshots
    at t=0. There is no temporal structure — the difference between input and
    output is spatial resolution, not time.

    Owns the dataset routing and input assembly decisions:
    - ``source_dataset``: the low-resolution input that gets upsampled
    - ``target_dataset``: the high-resolution output

    ``prepare_inputs`` upsamples ``source_dataset`` to the target grid via the
    model's ``InterpolationConnection`` before the forward pass, so the
    training method and model are agnostic about which datasets are involved.
    """

    name: str = "spatial-downscaler"

    def __init__(
        self,
        source_dataset: str = "in_lres",
        target_dataset: str = "out_hres",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def prepare_inputs(
        self,
        x: dict[str, torch.Tensor],
        model,
        model_comm_group=None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Upsample source_dataset (lres) to the target grid before the forward pass.

        Uses the model's InterpolationConnection so the upsampling matrix stays
        inside the model. The result replaces the raw lres tensor in x so the
        training method and model forward pass receive an already-upsampled input.
        """
        x_lres = x[self.source_dataset]
        x_upsampled = model.model.residual[self.source_dataset](
            x_lres,
            grid_shard_shapes=None,
            model_comm_group=model_comm_group,
        )[
            :,
            None,
            None,
            :,
            :,
        ]  # add time and ensemble dims: (batch, time=1, ensemble=1, grid, vars)
        return {**x, self.source_dataset: x_upsampled}
