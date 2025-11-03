# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from torch import nn

from anemoi.models.layers.block import PointWiseMLPProcessorBlock
from anemoi.models.layers.utils import load_layer_kernels

LOGGER = logging.getLogger(__name__)


class TestPointWiseMLPProcessorBlock:
    @given(
        num_channels=st.integers(min_value=1, max_value=64),
        mlp_hidden_ratio=st.integers(min_value=1, max_value=16),
        activation=st.sampled_from(
            [
                "torch.nn.ReLU",
                "torch.nn.GELU",
            ]
        ),
        dropout_p=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_init(self, num_channels, mlp_hidden_ratio, activation, dropout_p):
        hidden_dim = num_channels * mlp_hidden_ratio
        layer_kernels = load_layer_kernels({"Activation": {"_target_": activation}})

        block = PointWiseMLPProcessorBlock(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            layer_kernels=layer_kernels,
        )
        assert isinstance(block, PointWiseMLPProcessorBlock)

        assert isinstance(block.mlp, nn.Sequential)

    @given(
        num_channels=st.integers(min_value=1, max_value=64),
        mlp_hidden_ratio=st.integers(min_value=1, max_value=16),
        activation=st.sampled_from(
            [
                "torch.nn.ReLU",
                "torch.nn.GELU",
                "anemoi.models.layers.activations.GLU",
                "anemoi.models.layers.activations.SwiGLU",
            ]
        ),
        shapes=st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3),
        batch_size=st.integers(min_value=1, max_value=40),
        dropout_p=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_forward_output(
        self,
        num_channels,
        mlp_hidden_ratio,
        activation,
        shapes,
        batch_size,
        dropout_p,
    ):
        hidden_dim = num_channels * mlp_hidden_ratio
        kwargs = dict()
        if "GLU" in activation:
            kwargs["dim"] = hidden_dim
        layer_kernels = load_layer_kernels({"Activation": {"_target_": activation, **kwargs}})

        block = PointWiseMLPProcessorBlock(
            num_channels=num_channels,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p,
            layer_kernels=layer_kernels,
        )

        x = torch.randn((batch_size, num_channels))  # .to(torch.float16, non_blocking=True)
        output = block.forward(x, shapes, batch_size)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, num_channels)
