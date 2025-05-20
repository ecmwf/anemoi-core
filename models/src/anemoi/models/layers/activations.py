# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def leaky_hardtanh(
    input: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    negative_slope: float = 0.01,
    positive_slope: float = 0.01,
) -> torch.Tensor:
    """Leaky version of hardtanh where regions outside [min_val, max_val] have small non-zero slopes.

    Args:
        input: Input tensor
        min_val: Minimum value for the hardtanh region
        max_val: Maximum value for the hardtanh region
        negative_slope: Slope for values below min_val
        positive_slope: Slope for values above max_val

    Returns:
        Tensor with leaky hardtanh applied
    """
    below_min = input < min_val
    above_max = input > max_val
    # Standard hardtanh behavior for the middle region
    result = torch.clamp(input, min_val, max_val)
    # Add leaky behavior for regions outside the clamped range
    result = torch.where(below_min, min_val + negative_slope * (input - min_val), result)
    result = torch.where(above_max, max_val + positive_slope * (input - max_val), result)
    return result


class GLU(nn.Module):
    """Gated Linear Unit (GLU) layer.

    The GLU layer is a feedforward neural network layer that applies a gating
    mechanism to the input tensor. The layer is defined as:

    Equation: GLU(x, W, V, b, c) = σ(xW + b) ⊗ (xV + c)

    The GLU layer can be initialised with a variation activation function
    that is applied to the weights. You can pass in activation functions
    like nn.SiLU(), nn.ReLU(), etc. The default is nn.Sigmoid().

    In Hydra this can be initialised with the following config:

    ```yaml
    Activation:
      _target_: anemoi.models.layers.activations.GLU
      bias: False
      variation:
        _target_: torch.nn.ReLU6
    ```

    Note: In some preliminary tests, unbounded activations like ReLU, GeLU, or
    Swish but even LogSigmoid, have shown to numerically explode without using
    the projection layer. Bounded activations like Sigmoid, Tanh, or ReLU6
    seem to work well without the projection layer.

    Parameters
    ----------
    variation : nn.Module, optional
        The activation function to apply to the weights, by default nn.Sigmoid().
    bias : bool
        Whether to apply a bias term in the linear layers, by default True.
    projection : bool
        Whether to project the input tensor to a higher dimension, by default True.
    """

    def __init__(self, *, variation: Optional[nn.Module] = None, bias: bool = True, projection: bool = True):
        super().__init__()
        self.bias, self.projection = bias, projection
        self.dim = self.W = self.V = None
        self.variation = variation if variation is not None else nn.Sigmoid()

    def _post_init(self, x: torch.Tensor) -> None:
        """Initialize the weights and biases."""
        self.dim = x.shape[-1]
        if self.projection:
            self.projection = nn.Linear(self.dim, self.dim * 2, bias=False, device=x.device)
        self.V = nn.Linear(self.dim, self.dim, bias=self.bias, device=x.device)
        self.W = nn.Linear(self.dim, self.dim, bias=self.bias, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim is None:
            self._post_init(x)

        if self.projection:
            x, gate = self.projection(x).chunk(2, dim=-1)
        else:
            gate = x

        return self.variation(self.W(gate)) * self.V(x)


class SwiGLU(GLU):
    """SwiGLU activation function.

    The SwiGLU activation function is a combination of the Swish and GLU
    activation functions. It is defined as:

    Equation: SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)

    Parameters
    ----------
    bias : bool
        Whether to apply a bias term in the linear layers, by default True.
    """

    def __init__(self, *, bias: bool = True):
        super().__init__(variation=torch.nn.SiLU(), bias=bias, projection=True)


class GEGLU(GLU):
    """GEGLU activation function.

    The GEGLU activation function is a combination of the GELU and GLU
    activation functions. It is defined as:

    Equation: GEGLU(x, W, V, b, c) = GELU(xW + b) ⊗ (xV + c)

    Parameters
    ----------
    bias : bool
        Whether to apply a bias term in the linear layers, by default True.
    """

    def __init__(self, *, bias: bool = True):
        super().__init__(variation=torch.nn.GELU(), bias=bias, projection=True)


class ReGLU(GLU):
    """ReGLU activation function.

    The ReGLU activation function is a combination of the ReLU and GLU
    activation functions. It is defined as:

    Equation: ReGLU(x, W, V, b, c) = ReLU(xW + b) ⊗ (xV + c)

    Parameters
    ----------
    bias : bool
        Whether to apply a bias term in the linear layers, by default True.
    """

    def __init__(self, *, bias: bool = True):
        super().__init__(variation=torch.nn.ReLU(), bias=bias, projection=True)


class Sine(nn.Module):
    """Sine activation function.

    Periodic activation function defined as:

    Equation: Sine(x, w, phi) = sin(w x + phi)
    """

    def __init__(self, w=1, phi=0):
        super().__init__()
        self.w = w
        self.phi = phi

    def forward(self, x):
        return torch.sin(self.w * x + self.phi)
