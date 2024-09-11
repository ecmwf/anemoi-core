# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.mlp import MLP


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


class TestMLP:
    def test_init(self, num_features, hdim, num_out_feature):
        """Test MLP initialization."""
        mlp = MLP(num_features, hdim, num_out_feature, 0, "SiLU")
        assert isinstance(mlp, MLP)
        assert isinstance(mlp.model, torch.nn.Sequential)
        assert len(mlp.model) == 6

        mlp = MLP(num_features, hdim, num_out_feature, 0, "ReLU", False, False, False)
        assert len(mlp.model) == 5

        mlp = MLP(num_features, hdim, num_out_feature, 1, "SiLU", False, False, False)
        assert len(mlp.model) == 7

    def test_forwards(self, batch_size, nlatlon, num_features, hdim, num_out_feature):
        """Test MLP forward pass."""

        mlp = MLP(num_features, hdim, num_out_feature, layer_norm=True)
        x_in = torch.randn((batch_size, nlatlon, num_features), dtype=torch.float32, requires_grad=True)

        out = mlp(x_in)
        assert out.shape == (
            batch_size,
            nlatlon,
            num_out_feature,
        ), "Output shape is not correct"

    def test_backward(self, batch_size, nlatlon, num_features, hdim):
        """Test MLP backward pass."""

        x_in = torch.randn((batch_size, nlatlon, num_features), dtype=torch.float32, requires_grad=True)
        mlp_1 = MLP(num_features, hdim, hdim, layer_norm=True)

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
