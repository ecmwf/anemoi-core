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
from torch import nn

from anemoi.models.layers.graph import TrainableTensor


class TestTrainableTensor:
    @pytest.fixture
    def init(self):
        return 10, 5

    @pytest.fixture
    def trainable_tensor(self, init):
        return TrainableTensor(*init)

    @pytest.fixture
    def x(self, init):
        size = init[0]
        return torch.rand(size, size)

    def test_init(self, trainable_tensor):
        assert isinstance(trainable_tensor, TrainableTensor)
        assert isinstance(trainable_tensor.trainable, nn.Parameter)

    def test_forward_backward(self, init, trainable_tensor, x):
        batch_size = 5
        output = trainable_tensor(x, batch_size)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size * x.shape[0], sum(init))

        # Dummy loss
        target = torch.rand(output.shape)
        loss_fn = nn.MSELoss()

        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        for param in trainable_tensor.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_forward_no_trainable(self, init, x):
        tensor_size, trainable_size = init[0], 0

        trainable_tensor = TrainableTensor(tensor_size, trainable_size)
        assert trainable_tensor.trainable is None

        batch_size = 5
        output = trainable_tensor(x, batch_size)
        assert output.shape == (batch_size * x.shape[0], tensor_size + trainable_size)
