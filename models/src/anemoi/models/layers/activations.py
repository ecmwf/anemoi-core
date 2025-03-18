# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch import nn


class GLU(nn.Module):
    """Gated Linear Unit (GLU) layer.

    The GLU layer is a feedforward neural network layer that applies a gating
    mechanism to the input tensor. The layer is defined as:

    Equation: GLU(x, W, V, b, c) = σ(xW + b) ⊗ (xV + c)

    The GLU layer can be initialised with a variation activation function
    that is applied to the weights.

    Parameters
    ----------
    variation : nn.Module, optional
        The activation function to apply to the weights, by default
        nn.Sigmoid().
    """

    def __init__(self, variation: nn.Module | None = None):
        super().__init__()
        self.dim = self.W = self.V = None
        self.variation = variation if variation is not None else nn.Sigmoid()

    def _post_init(self, x: torch.Tensor) -> None:
        """Initialize the weights and biases."""
        self.dim = x.shape[-1]
        self.W = nn.Linear(self.dim, self.dim).to(x.device)
        self.V = nn.Linear(self.dim, self.dim).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            self._post_init(x)

        return self.variation(self.W(x)) * self.V(x)


class SwiGLU(GLU):
    """Swish GLU layer."""

    def __init__(self):
        super().__init__(nn.SiLU())


class ReGLU(GLU):
    """ReLU GLU layer."""

    def __init__(self):
        super().__init__(nn.ReLU())


class GeGLU(GLU):
    """GELU GLU layer."""

    def __init__(self):
        super().__init__(nn.GELU())


class Sine(nn.Module):
    """Sine activation function.

    Periodic activation function defined as:

    Equation: Sine(x, w, phi) = sin(w x + phi)
    """
    def __init__(self, w = 1, phi = 0):
        super().__init__()
        self.w = w
        self.phi = phi

    def forward(self, x):
        return torch.sin(self.w * x + self.phi)
