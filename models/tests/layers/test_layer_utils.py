# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
import torch.nn as nn
from hydra.errors import InstantiationException
from hypothesis import given
from hypothesis import strategies as st
from omegaconf import OmegaConf

from anemoi.models.layers.utils import CheckpointWrapper
from anemoi.models.layers.utils import load_layer_kernels


class TestLayerUtils:
    """Test the layer utility functions."""

    @pytest.fixture(scope="class")
    def default_layer_kernels(self):
        return load_layer_kernels(None)

    @given(in_features=st.integers(min_value=1, max_value=100), out_features=st.integers(min_value=1, max_value=100))
    def test_default_kernels_init(self, default_layer_kernels, in_features, out_features):
        linear_layer = default_layer_kernels.Linear(in_features=in_features, out_features=out_features)
        layer_norm = default_layer_kernels.LayerNorm(normalized_shape=out_features)
        activation = default_layer_kernels.Activation()

        assert isinstance(linear_layer, nn.Linear)
        assert isinstance(layer_norm, nn.LayerNorm)
        assert isinstance(activation, nn.GELU)

        assert linear_layer.in_features == in_features
        assert linear_layer.out_features == out_features
        assert linear_layer.bias.shape == torch.Size([out_features])
        assert layer_norm.normalized_shape == (out_features,)

    @given(eps=st.floats(min_value=1e-6, max_value=1e-3), elementwise_affine=st.booleans(), bias=st.booleans())
    def test_custom_kernels(self, eps, elementwise_affine, bias):
        kernels_config = OmegaConf.create(
            {
                "LayerNorm": {
                    "_target_": "torch.nn.LayerNorm",
                    "eps": eps,
                    "elementwise_affine": elementwise_affine,
                },
                "Linear": {"_target_": "torch.nn.Linear", "bias": bias},
                "Activation": {"_target_": "torch.nn.ReLU"},
            }
        )

        custom_kernels = load_layer_kernels(kernels_config)

        linear_layer = custom_kernels.Linear(in_features=10, out_features=10)
        layer_norm = custom_kernels.LayerNorm(normalized_shape=10)
        activation = custom_kernels.Activation()

        assert isinstance(linear_layer, nn.Linear)
        assert isinstance(layer_norm, nn.LayerNorm)
        assert isinstance(activation, nn.ReLU)

        if bias:
            assert linear_layer.bias is not None
        else:
            assert linear_layer.bias is None

        assert abs(layer_norm.eps - eps) < 1e-10
        assert layer_norm.elementwise_affine == elementwise_affine

    def test_unavailable_kernel(self):
        kernels_config = OmegaConf.create(
            {
                "LayerNorm": {"_target_": "nonexistent_package.LayerNorm"},
            }
        )

        with pytest.raises(InstantiationException):
            load_layer_kernels(kernels_config)

    @given(input_shape=st.lists(st.integers(1, 20), min_size=2, max_size=4))
    def test_kernel_forward_pass(self, default_layer_kernels, input_shape):
        # Create random input
        hidden_dim = input_shape[-1]
        x = torch.rand(*input_shape)

        # Initialize layers
        linear = default_layer_kernels.Linear(hidden_dim, hidden_dim)
        layer_norm = default_layer_kernels.LayerNorm(hidden_dim)
        activation = default_layer_kernels.Activation()

        # Forward pass
        output = activation(layer_norm(linear(x)))

        # Check output shape
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestCheckpointWrapper:
    """Test the CheckpointWrapper utility."""

    class SimpleModule(nn.Module):
        """Dummy module with a single linear layer to wrap."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    @given(batch_size=st.integers(1, 10))
    def test_checkpoint_wrapper(self, batch_size):
        module = self.SimpleModule()
        wrapped_module = CheckpointWrapper(module)

        # Test with random input
        x = torch.rand(batch_size, 10)

        # Forward pass through both modules
        with torch.no_grad():
            regular_output = module(x)
            checkpoint_output = wrapped_module(x)

        # Outputs should be identical
        assert torch.allclose(regular_output, checkpoint_output)

    def test_checkpoint_wrapper_gradient(self):
        module = self.SimpleModule()
        wrapped_module = CheckpointWrapper(module)

        # Input requires gradient
        x = torch.rand(5, 10, requires_grad=True)

        # Forward and backward pass
        regular_output = module(x).sum()
        regular_output.backward()
        regular_grad = x.grad.clone()

        # Reset gradients
        x.grad = None

        # Forward and backward with checkpoint
        checkpoint_output = wrapped_module(x).sum()
        checkpoint_output.backward()
        checkpoint_grad = x.grad.clone()

        # Gradients should be identical
        assert torch.allclose(regular_grad, checkpoint_grad)
