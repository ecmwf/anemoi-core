# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Ensemble training method with data-assimilation cycling and correctors.

Pairs with :class:`anemoi.training.tasks.da_forecaster.DAForecaster` and an
:class:`anemoi.models.models.AnemoiEnsModelEncProcDec`, scoring the members with
kernel CRPS.

The DA cycling, correctors, ``da_loss_weight`` weighting and DATA_OUTPUT target
slicing come from :class:`~anemoi.training.train.methods.da_mixin.DATrainingMixin`;
the member dimension, the ensemble communication groups, the differentiable member
allgather before the loss and the ensemble plot adapter come from
:class:`~anemoi.training.train.methods.ensemble.EnsembleTraining`.

The corrector therefore runs on the members held locally, before the allgather.

This class deliberately defines no ``__init__``: ``EnsembleTraining`` already reads
``training.ensemble_size_per_device`` from the config, and an extra ``__init__``
frame would change what Lightning's ``save_hyperparameters`` records.
"""

import torch

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.train.methods.da_mixin import DATrainingMixin
from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.utils.enums import TensorDim


class DAEnsembleTraining(DATrainingMixin, EnsembleTraining):
    """Ensemble training method for the DA forecaster with correctors."""

    def _prepare_state(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Tile the initial state to the members held on this device."""
        return self._expand_ens_dim(x)

    def _step_forward_kwargs(self, task_kwargs: dict) -> dict:
        """Pass the step index so ``EnsembleTraining.forward`` can set ``fcstep``.

        ``rollout_step`` is continuous across the DA cycles and the forecast, and the
        ensemble model clamps it to ``min(1, fcstep)``, so only the first DA cycle is
        conditioned as the initial step.
        """
        return {"rollout_step": task_kwargs["rollout_step"]}

    def _output_prediction(
        self,
        y_pred: dict[str, torch.Tensor],
        validation_mode: bool,
    ) -> dict[str, torch.Tensor]:
        """Gather the members of the ensemble subgroup into the returned prediction.

        Each device holds only ``ensemble_size_per_device`` members. The plot callbacks
        score the prediction with the training loss, and kernel CRPS needs more than one
        member, so they need the members from the whole subgroup, as
        ``EnsembleTraining._step`` returns them.

        Only validation outputs are consumed (the plot and per-timestep metric callbacks
        run on validation batches), so training steps skip the extra collective. Every
        rank runs the same task steps, so the gather is entered by all of them.
        """
        if not validation_mode:
            return y_pred
        with torch.no_grad():
            return {
                name: gather_tensor(
                    pred,
                    dim=TensorDim.ENSEMBLE_DIM,
                    sizes=[pred.size(TensorDim.ENSEMBLE_DIM)] * self.ens_comm_subgroup_size,
                    mgroup=self.ens_comm_subgroup,
                )
                for name, pred in y_pred.items()
            }
