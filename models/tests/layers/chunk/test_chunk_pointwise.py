# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.models.layers.block import PointWiseMLPProcessorBlock
from anemoi.models.layers.chunk import PointWiseMLPProcessorChunk
from anemoi.models.layers.utils import load_layer_kernels


class TestPointWiseMLPProcessorChunk:
    @pytest.fixture
    def init(self):
        num_channels = 512
        num_layers = 3
        mlp_hidden_ratio: int = 4
        dropout_p: float = 0.1
        layer_kernels = load_layer_kernels()

        return (
            num_channels,
            num_layers,
            layer_kernels,
            mlp_hidden_ratio,
            dropout_p,
        )

    @pytest.fixture
    def processor_chunk(self, init):
        (
            num_channels,
            num_layers,
            layer_kernels,
            mlp_hidden_ratio,
            dropout_p,
        ) = init
        return PointWiseMLPProcessorChunk(
            num_channels=num_channels,
            num_layers=num_layers,
            layer_kernels=layer_kernels,
            mlp_hidden_ratio=mlp_hidden_ratio,
            dropout_p=dropout_p,
        )

    def test_all_blocks(self, processor_chunk):
        assert all(isinstance(block, PointWiseMLPProcessorBlock) for block in processor_chunk.blocks)
