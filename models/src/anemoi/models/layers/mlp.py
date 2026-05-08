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
from torch import nn

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron with optional checkpoint."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        layer_kernels: DotDict,
        n_extra_layers: int = 0,
        final_activation: bool = False,
        layer_norm: bool = True,
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
        final_activation : bool, optional
            Whether to apply a final activation function to last layer, by default True
        layer_norm : bool, optional
            Whether to apply layer norm after activation, by default True
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml

        Returns
        -------
        nn.Module
            Returns a MLP module
        """
        super().__init__()

        Linear = layer_kernels.Linear
        LayerNorm = layer_kernels.LayerNorm
        Activation = layer_kernels.Activation

        self.mlp = nn.Sequential(Linear(in_features, hidden_dim), Activation())
        for _ in range(n_extra_layers + 1):
            self.mlp.append(Linear(hidden_dim, hidden_dim))
            self.mlp.append(Activation())
        self.mlp.append(Linear(hidden_dim, out_features))

        if final_activation:
            self.mlp.append(Activation())

        self.layer_norm = None
        if layer_norm:
            self.layer_norm = LayerNorm(normalized_shape=out_features)

    def forward(self, x: torch.Tensor, **layer_kwargs) -> torch.Tensor:
        x = self.mlp(x)
        if self.layer_norm:
            x = self.layer_norm(x, **layer_kwargs)
        return x
