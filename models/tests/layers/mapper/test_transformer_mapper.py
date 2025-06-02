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
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.layers.mapper import TransformerBaseMapper
from anemoi.models.layers.utils import load_layer_kernels


class TestTransformerBaseMapper:
    """Test the GraphTransformerBaseMapper class."""

    NUM_EDGES: int = 150
    NUM_SRC_NODES: int = 100
    NUM_DST_NODES: int = 200

    @pytest.fixture
    def layer_kernels(self):
        kernel_config = OmegaConf.create(
            {
                "LayerNorm": {
                    "_target_": "torch.nn.LayerNorm",
                },
                "Linear": {"_target_": "torch.nn.Linear", "bias": False},
            }
        )
        return load_layer_kernels(kernel_config, instance=False)

    @pytest.fixture
    def mapper_init(self, layer_kernels):
        in_channels_src: int = 3
        in_channels_dst: int = 3
        hidden_dim: int = 256
        out_channels_dst: int = 5
        num_chunks: int = 1
        cpu_offload: bool = False
        num_heads: int = 16
        mlp_hidden_ratio: int = 7
        attention_implementation: str = "scaled_dot_product_attention"
        return (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            num_chunks,
            cpu_offload,
            num_heads,
            mlp_hidden_ratio,
            layer_kernels,
            attention_implementation,
        )

    @pytest.fixture
    def mapper(self, mapper_init):
        (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            num_chunks,
            cpu_offload,
            num_heads,
            mlp_hidden_ratio,
            layer_kernels,
            attention_implementation,
        ) = mapper_init
        return TransformerBaseMapper(
            in_channels_src=in_channels_src,
            in_channels_dst=in_channels_dst,
            hidden_dim=hidden_dim,
            layer_kernels=layer_kernels,
            out_channels_dst=out_channels_dst,
            num_chunks=num_chunks,
            cpu_offload=cpu_offload,
            num_heads=num_heads,
            mlp_hidden_ratio=mlp_hidden_ratio,
            attention_implementation=attention_implementation,
        )

    @pytest.fixture
    def pair_tensor(self, mapper_init):
        (
            in_channels_src,
            in_channels_dst,
            _hidden_dim,
            _out_channels_dst,
            _num_chunks,
            _cpu_offload,
            _num_heads,
            _mlp_hidden_ratio,
            _layer_kernels,
            _attention_implementation,
        ) = mapper_init
        return (
            torch.rand(self.NUM_SRC_NODES, in_channels_src),
            torch.rand(self.NUM_DST_NODES, in_channels_dst),
        )

    @pytest.fixture
    def fake_graph(self) -> HeteroData:
        """Fake graph."""
        graph = HeteroData()
        graph[("src", "to", "dst")].edge_index = torch.concat(
            [
                torch.randint(0, self.NUM_SRC_NODES, (1, self.NUM_EDGES)),
                torch.randint(0, self.NUM_DST_NODES, (1, self.NUM_EDGES)),
            ],
            axis=0,
        )
        graph[("src", "to", "dst")].edge_attr1 = torch.rand((self.NUM_EDGES, 1))
        graph[("src", "to", "dst")].edge_attr2 = torch.rand((self.NUM_EDGES, 32))
        return graph

    def test_initialization(self, mapper, mapper_init):
        (
            in_channels_src,
            in_channels_dst,
            hidden_dim,
            out_channels_dst,
            _num_chunks,
            _cpu_offload,
            _num_heads,
            _mlp_hidden_ratio,
            _layer_kernels,
            _attention_implementation,
        ) = mapper_init
        assert isinstance(mapper, TransformerBaseMapper)
        assert mapper.in_channels_src == in_channels_src
        assert mapper.in_channels_dst == in_channels_dst
        assert mapper.hidden_dim == hidden_dim
        assert mapper.out_channels_dst == out_channels_dst

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        # Should be a no-op in the base class
        x = pair_tensor
        (
            _in_channels_src,
            _in_channels_dst,
            _hidden_dim,
            _out_channels_dst,
            _num_chunks,
            _cpu_offload,
            _num_heads,
            _mlp_hidden_ratio,
            _layer_kernels,
            _attention_implementation,
        ) = mapper_init
        shard_shapes = [list(x[0].shape)], [list(x[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(x, shard_shapes)
        assert x_src.shape == torch.Size(
            x[0].shape
        ), f"x_src.shape ({x_src.shape}) != torch.Size(x[0].shape) ({torch.Size(x[0].shape)})"
        assert x_dst.shape == torch.Size(
            x[1].shape
        ), f"x_dst.shape ({x_dst.shape}) != torch.Size(x[1].shape) ({x[1].shape})"
        assert shapes_src == [
            list(x[0].shape)
        ], f"shapes_src ({shapes_src}) != [list(x[0].shape)] ({[list(x[0].shape)]})"
        assert shapes_dst == [
            list(x[1].shape)
        ], f"shapes_dst ({shapes_dst}) != [list(x[1].shape)] ({[list(x[1].shape)]})"

    def test_post_process(self, mapper, pair_tensor):
        # Should be a no-op in the base class
        x_dst = pair_tensor[1]
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(x_dst, shapes_dst)
        assert torch.equal(result, x_dst)
