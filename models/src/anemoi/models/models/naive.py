# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Naive baseline model: per-gridpoint linear layer, no graph operations."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.utils.config import DotDict

from anemoi.models.interface import AnemoiModelInterface


class NaiveModel(AnemoiModelInterface):
    """Simplest possible baseline: independent linear layer per grid point.

    No graph neural network, no encoder/processor/decoder — just a learned
    linear map from (n_step_input × n_input_vars) to (n_step_output × n_output_vars)
    applied identically at every grid point.

    Implements AnemoiModelInterface directly, independent of AnemoiModel.
    """

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        metadata: dict,
        supporting_arrays: dict | None = None,
        **_,
    ) -> None:
        super().__init__()

        self.config = model_config
        self.metadata = metadata
        self.supporting_arrays = supporting_arrays or {}
        self.data_indices = data_indices

        n_step_input = model_config.training.multistep_input
        n_step_output = model_config.training.multistep_output
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output
        self._n_output_vars = {name: len(idx.model.output) for name, idx in data_indices.items()}

        self.linear = nn.ModuleDict({
            name: nn.Linear(
                len(idx.model.input) * n_step_input,
                len(idx.model.output) * n_step_output,
            )
            for name, idx in data_indices.items()
        })

    def forward(self, x: dict[str, Tensor], **_) -> dict[str, Tensor]:
        out = {}
        for name, x_ds in x.items():
            bs, t, ens, grid, nv = x_ds.shape
            x_flat = x_ds.reshape(bs * grid, t * nv)
            y_flat = self.linear[name](x_flat)
            out[name] = y_flat.reshape(bs, self.n_step_output, ens, grid, self._n_output_vars[name])
        return out

    def predict_step(
        self,
        batch: dict[str, Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            x = {name: batch[name][:, : self.n_step_input, None, ...] for name in batch}
            return self.forward(x)
