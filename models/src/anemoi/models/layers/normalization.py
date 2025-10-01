# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Union

from torch import Size
from torch import Tensor
from torch import nn


class AutocastLayerNorm(nn.LayerNorm):
    """LayerNorm that casts the output back to the input type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with explicit autocast back to the input type.

        This casts the output to (b)float16 (instead of float32) when we run in mixed
        precision.
        """
        return super().forward(x).type_as(x)


class ConditionalLayerNorm(nn.Module):
    """Conditional Layer Normalization.

    x_norm = a(u) * (x - mean) / sqrt(var + eps) + b(u)

    """

    def __init__(
        self,
        normalized_shape: Union[int, list, Size],
        condition_shape: int = 16,
        w_one_bias_zero_init: bool = True,
        autocast: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)  # no learnable parameters
        self.scale = nn.Linear(condition_shape, normalized_shape)  # , bias=False)
        self.bias = nn.Linear(condition_shape, normalized_shape)  # , bias=False)
        self.autocast = autocast

        if w_one_bias_zero_init:
            nn.init.zeros_(self.scale.weight)
            nn.init.zeros_(self.scale.bias)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Conditional Layer Normalization.

        Parameters
        ----------
        input : List[Tensor, Tensor]
            A list of two tensors (x, cond),
            the first is the input tensor and
            the second is the condition tensor.

        Returns
        -------
        Tensor
            The output tensor.
        """
        scale = self.scale(cond)
        bias = self.bias(cond)
        out = self.norm(x)
        out = out * (scale + 1.0) + bias
        return out.type_as(x) if self.autocast else out
