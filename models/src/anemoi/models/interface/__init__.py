# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import torch
from torch.distributed.distributed_c10d import ProcessGroup


@runtime_checkable
class ModelInterface(Protocol):
    """Interface for Anemoi models."""

    def pre_process(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply pre-processing (e.g. normalisation) to model inputs."""
        ...

    def post_process(self, y: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply post-processing (e.g. denormalisation) to model outputs."""
        ...

    def forward(
        self,
        x: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run the model forward pass on pre-processed inputs."""
        ...

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run a full inference step (pre-process → forward → post-process)."""
        ...


__all__ = ["ModelInterface"]
