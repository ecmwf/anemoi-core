# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import TYPE_CHECKING

from anemoi.training.train.methods.single import SingleTraining

if TYPE_CHECKING:
    import torch


class LatentRolloutTraining(SingleTraining):
    """Deterministic training that passes the rollout step to the model.

    Models such as ``AnemoiModelDisentangledEncProcDec`` with
    ``model.latent_rollout: True`` use ``rollout_step`` to advance their latent
    state instead of re-encoding the input at every rollout step. Tasks without
    rollout (e.g. autoencoding) call the model with ``rollout_step=0``.
    """

    def _forward_step(
        self,
        x: dict[str, torch.Tensor],
        rollout_step: int = 0,
        **_task_kwargs,
    ) -> dict[str, torch.Tensor]:
        return self(x, rollout_step=rollout_step)
