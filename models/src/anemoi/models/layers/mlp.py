# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Literal

import torch
from torch import nn

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

MLPImplementation = Literal["mlp", "glu", "swiglu", "geglu", "reglu"]


def _build_gating_activation(mlp_implementation: MLPImplementation) -> nn.Module:
    if mlp_implementation == "glu":
        return nn.Sigmoid()
    if mlp_implementation == "swiglu":
        return nn.SiLU()
    if mlp_implementation == "geglu":
        return nn.GELU()
    if mlp_implementation == "reglu":
        return nn.ReLU()
    valid = ("mlp", "glu", "swiglu", "geglu", "reglu")
    raise ValueError(f"`mlp_implementation` must be one of {valid}, got '{mlp_implementation}'.")


class GatedMLPLayer(nn.Module):
    """Single gated feed-forward layer used by GLU variants."""

    def __init__(
        self, in_features: int, out_features: int, layer_kernels: DotDict, mlp_implementation: MLPImplementation
    ):
        super().__init__()
        Linear = layer_kernels.Linear
        self.gate_proj = Linear(in_features, out_features)
        self.value_proj = Linear(in_features, out_features)
        self.gating = _build_gating_activation(mlp_implementation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gating(self.gate_proj(x)) * self.value_proj(x)


def build_feedforward_layer(
    in_features: int,
    out_features: int,
    layer_kernels: DotDict,
    mlp_implementation: MLPImplementation = "mlp",
) -> nn.Module:
    """Build one feed-forward layer module."""
    modules = build_feedforward_modules(
        in_features=in_features,
        out_features=out_features,
        layer_kernels=layer_kernels,
        mlp_implementation=mlp_implementation,
    )
    return modules[0] if len(modules) == 1 else nn.Sequential(*modules)


def build_feedforward_modules(
    in_features: int,
    out_features: int,
    layer_kernels: DotDict,
    mlp_implementation: MLPImplementation = "mlp",
) -> list[nn.Module]:
    """Build one feed-forward layer as a flat list of modules."""
    Linear = layer_kernels.Linear

    if mlp_implementation == "mlp":
        activation = layer_kernels.Activation()
        if "GLU" in activation.__class__.__name__.upper():
            raise ValueError(
                "GLU-based activations are not supported via layer_kernels.Activation. "
                "Use `mlp_implementation` with one of: 'glu', 'swiglu', 'geglu', 'reglu'."
            )
        return [Linear(in_features, out_features), activation]

    return [GatedMLPLayer(in_features, out_features, layer_kernels, mlp_implementation)]


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
        mlp_implementation: MLPImplementation = "mlp",
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
        mlp_implementation : MLPImplementation, optional
            Implementation of hidden feed-forward layers: `mlp`, `glu`, `swiglu`, `geglu`, or `reglu`.
        layer_kernels : DotDict
            A dict of layer implementations e.g. layer_kernels.Linear = "torch.nn.Linear"
            Defined in config/models/<model>.yaml

        Returns
        -------
        nn.Module
            Returns a MLP module
        """
        super().__init__()
        if n_extra_layers < 0:
            msg = f"`n_extra_layers` must be >= 0, got {n_extra_layers}."
            raise ValueError(msg)

        Linear = layer_kernels.Linear
        LayerNorm = layer_kernels.LayerNorm

        layers: list[nn.Module] = []

        def _append_flat_ffn_layer(in_dim: int, out_dim: int) -> None:
            layers.extend(
                build_feedforward_modules(
                    in_features=in_dim,
                    out_features=out_dim,
                    layer_kernels=layer_kernels,
                    mlp_implementation=mlp_implementation,
                )
            )

        _append_flat_ffn_layer(in_features, hidden_dim)
        for _ in range(n_extra_layers):
            _append_flat_ffn_layer(hidden_dim, hidden_dim)
        layers.append(Linear(hidden_dim, out_features))

        if final_activation:
            if mlp_implementation == "mlp":
                layers.append(layer_kernels.Activation())
            else:
                layers.append(_build_gating_activation(mlp_implementation))

        self.mlp = nn.Sequential(*layers)

        self.layer_norm = None
        if layer_norm:
            self.layer_norm = LayerNorm(normalized_shape=out_features)

    def forward(self, x: torch.Tensor, **layer_kwargs) -> torch.Tensor:
        x = self.mlp(x)
        if self.layer_norm:
            x = self.layer_norm(x, **layer_kwargs)
        return x
