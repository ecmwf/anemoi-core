# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch

from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.utils import load_layer_kernels


@pytest.fixture
def batch_size():
    return 1


@pytest.fixture
def nlatlon():
    return 1024


@pytest.fixture
def num_features():
    return 64


@pytest.fixture
def hdim():
    return 128


@pytest.fixture
def num_out_feature():
    return 36


@pytest.fixture
def layer_kernels():
    return load_layer_kernels()


class TestMLP:
    def test_init(self, num_features, hdim, num_out_feature, layer_kernels):
        """Test MLP initialization."""
        mlp = MLP(num_features, hdim, num_out_feature, layer_kernels, n_extra_layers=0)
        assert isinstance(mlp, MLP)
        assert isinstance(mlp.mlp, torch.nn.Sequential)
        assert len(mlp.mlp) == 5

        mlp = MLP(num_features, hdim, num_out_feature, layer_kernels, 0, False, False)
        assert len(mlp.mlp) == 5

        mlp = MLP(num_features, hdim, num_out_feature, layer_kernels, 1, False, False)
        assert len(mlp.mlp) == 7

    def test_forwards(self, batch_size, nlatlon, num_features, hdim, num_out_feature, layer_kernels):
        """Test MLP forward pass."""

        mlp = MLP(num_features, hdim, num_out_feature, layer_kernels=layer_kernels, layer_norm=True)
        x_in = torch.randn((batch_size, nlatlon, num_features), dtype=torch.float32, requires_grad=True)

        out = mlp(x_in)
        assert out.shape == (
            batch_size,
            nlatlon,
            num_out_feature,
        ), "Output shape is not correct"

    def test_backward(self, batch_size, nlatlon, num_features, hdim, layer_kernels):
        """Test MLP backward pass."""

        x_in = torch.randn((batch_size, nlatlon, num_features), dtype=torch.float32, requires_grad=True)
        mlp_1 = MLP(num_features, hdim, hdim, layer_kernels, layer_norm=True)

        y = mlp_1(x_in)
        assert y.shape == (batch_size, nlatlon, hdim)

        loss = y.sum()
        print("running backward on the dummy loss ...")
        loss.backward()

        for param in mlp_1.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"
