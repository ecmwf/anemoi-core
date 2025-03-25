# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.chunk import GNNProcessorChunk
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.utils import load_layer_kernels


class TestGNNProcessorChunk:
    @pytest.fixture
    def init(self):
        num_channels = 10
        num_layers = 3
        mlp_extra_layers = 3
        edge_dim = None
        layer_kernels = load_layer_kernels()
        return num_channels, num_layers, layer_kernels, mlp_extra_layers, edge_dim

    @pytest.fixture
    def processor_chunk(self, init):
        num_channels, num_layers, layer_kernels, mlp_extra_layers, edge_dim = init
        return GNNProcessorChunk(
            num_channels=num_channels,
            num_layers=num_layers,
            layer_kernels=layer_kernels,
            mlp_extra_layers=mlp_extra_layers,
            edge_dim=edge_dim,
        )

    def test_embed_edges(self, init, processor_chunk):
        _num_channels, _num_layers, _layer_kernels, _mlp_extra_layers, edge_dim = init
        if edge_dim:
            assert isinstance(processor_chunk.emb_edges, MLP)
        else:
            assert processor_chunk.emb_edges is None

    def test_all_blocks(self, processor_chunk):
        assert all(isinstance(block, GraphConvProcessorBlock) for block in processor_chunk.blocks)
