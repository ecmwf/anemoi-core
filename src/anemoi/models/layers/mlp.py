# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging

import torch
from torch import nn

from anemoi.models.layers.utils import AutocastLayerNorm
from anemoi.models.layers.utils import CheckpointWrapper

LOGGER = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron with optional checkpoint."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_extra_layers: int = 0,
        activation: str = "SiLU",
        final_activation: bool = False,
        layer_norm: bool = True,
        checkpoints: bool = False,
    ) -> nn.Module:
        """Generate a multi-layer perceptron.

        Parameters
        ----------
        in_features : int
            Number of input features
        hidden_dim : int
            Hidden dimensions
        out_features : int
            Number of output features
        n_extra_layers : int, optional
            Number of extra layers in MLP, by default 0
        activation : str, optional
            Activation function, by default "SiLU"
        final_activation : bool, optional
            Whether to apply a final activation function to last layer, by default True
        layer_norm : bool, optional
            Whether to apply layer norm after activation, by default True
        checkpoints : bool, optional
            Whether to provide checkpoints, by default False

        Returns
        -------
        nn.Module
            Returns a MLP module

        Raises
        ------
        RuntimeError
            If activation function is not supported
        """
        super().__init__()
        try:
            act_func = getattr(nn, activation)
        except AttributeError as ae:
            LOGGER.error("Activation function %s not supported", activation)
            raise RuntimeError from ae

        mlp1 = nn.Sequential(nn.Linear(in_features, hidden_dim), act_func())
        for _ in range(n_extra_layers + 1):
            mlp1.append(nn.Linear(hidden_dim, hidden_dim))
            mlp1.append(act_func())
        mlp1.append(nn.Linear(hidden_dim, out_features))

        if final_activation:
            mlp1.append(act_func())

        if layer_norm:
            mlp1.append(AutocastLayerNorm(out_features))

        self.model = CheckpointWrapper(mlp1) if checkpoints else mlp1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
