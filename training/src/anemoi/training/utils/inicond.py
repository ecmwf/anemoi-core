# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from omegaconf import DictConfig
from torch import nn

LOGGER = logging.getLogger(__name__)


class EnsembleInitialConditions(nn.Module):
    """Generates initial conditions for ensemble runs.

    Uses analysis and (optionally) EDA member data. This module has no buffers or
    trainable parameters.
    """

    def __init__(self, config: DictConfig, data_indices: dict) -> None:
        """Initialise object.

        Parameters
        ----------
        data_indices : dict
            Indices of the training data
        """
        super().__init__()

        self.data_indices = data_indices
        self.multi_step = config.training.multistep_input
        self.nens_per_device = config.training.ensemble_size_per_device
        self._q_indices = self._compute_q_indices()

    def _compute_q_indices(self) -> torch.Tensor | None:
        """Returns indices of humidity variables in input tensors.

        This step will later be included in the zarr building process.
        """
        q_idx = []
        for vname, vidx in self.data_indices.data.input.name_to_index.items():
            if vname.startswith("q_"):  # humidity (pl / ml)
                q_idx.append(vidx)
        LOGGER.debug("q_* indices in the input tensor: %s", 'q_idx if q_idx else "n/a"')
        return torch.IntTensor(q_idx) if q_idx else None

    def forward(self, x_an: torch.Tensor, x_eda: torch.Tensor | None = None) -> torch.Tensor:
        """Generate initial conditions for the ensemble based on the EDA perturbations.

        If no EDA perturbations are given, we simply stack the deterministic ERA5
        analysis nens_per_device times along the ensemble dimension.

        Inputs:
            x_an: unperturbed IC (ERA5 analysis), shape = (bs, ms + rollout, latlon, v)
            x_eda (optional): ERA5 EDA perturbations, shape = (bs, ms, nens_per_device, latlon, v)

        Returns
        -------
            Ensemble IC, shape (bs, ms, nens_per_device, latlon, input.full)
        """
        if x_eda is None:
            # no EDA available, just stack the analysis nens_per_device times along an ensemble dimension

            LOGGER.debug("NO EDA -- SHAPES: x_an.shape = %s, multi_step = %d", list(x_an.shape), self.multi_step)
            x_ = x_an[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices.data.input.full,
            ]  # (bs, ms, ens_dummy, latlon, nvar)

            return torch.cat([x_] * self.nens_per_device, dim=2)  # shape == (bs, ms, nens_per_device, latlon, nvar)

        LOGGER.debug("EDA -- SHAPES: x_an.shape = %s, x_eda.shape = %s", list(x_an.shape), list(x_eda.shape))
        assert x_eda.shape[1] == self.multi_step and x_eda.shape[2] == self.nens_per_device

        return x_eda[..., self.data_indices.data.input.full]
