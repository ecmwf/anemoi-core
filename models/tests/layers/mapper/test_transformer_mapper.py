# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import pytest
import torch

from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerBaseMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class MapperConfig:
    in_channels_src: int = 3
    in_channels_dst: int = 4
    hidden_dim: int = 128
    num_chunks: int = 2
    num_heads: int = 8
    mlp_hidden_ratio: int = 4
    attn_channels: Optional[int] = None
    qk_norm: bool = True
    dropout_p: float = 0.0
    attention_implementation: str = "scaled_dot_product_attention"
    softcap: Optional[float] = None
    use_alibi_slopes: bool = False
    cpu_offload: bool = False
    window_size: Optional[int] = None
    use_rotary_embeddings: bool = False
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


class ConcreteTransformerBaseMapper(TransformerBaseMapper):
    """Concrete implementation of TransformerBaseMapper for testing."""

    def pre_process(self, x, shard_shapes, model_comm_group=None, x_src_is_sharded=False, x_dst_is_sharded=False):
        shapes_src, shapes_dst = shard_shapes
        x_src, x_dst = x
        return x_src, x_dst, shapes_src, shapes_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst


class TestTransformerBaseMapper:
    NUM_SRC_NODES: int = 10
    NUM_DST_NODES: int = 12
    OUT_CHANNELS_DST: int = 5

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig()

    @pytest.fixture
    def mapper(self, mapper_init):
        return ConcreteTransformerBaseMapper(
            **asdict(mapper_init),
            out_channels_dst=self.OUT_CHANNELS_DST,
        )

    @pytest.fixture
    def pair_tensor(self, mapper_init):
        return (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst),
        )

    def test_initialization(self, mapper, mapper_init):
        assert isinstance(mapper, TransformerBaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.in_channels_dst == mapper_init.in_channels_dst
        assert mapper.hidden_dim == mapper_init.hidden_dim
        assert mapper.out_channels_dst == self.OUT_CHANNELS_DST

    def test_pre_process(self, mapper, pair_tensor):
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]

        x_src, x_dst, shapes_src, shapes_dst = mapper.pre_process(pair_tensor, shard_shapes)
        assert x_src.shape == torch.Size(pair_tensor[0].shape)
        assert x_dst.shape == torch.Size(pair_tensor[1].shape)
        assert shapes_src == [list(pair_tensor[0].shape)]
        assert shapes_dst == [list(pair_tensor[1].shape)]

    def test_post_process(self, mapper, pair_tensor):
        x_dst = pair_tensor[1]
        shapes_dst = [list(x_dst.shape)]

        result = mapper.post_process(x_dst, shapes_dst=shapes_dst)
        assert torch.equal(result, x_dst)


class TestTransformerForwardMapper:
    NUM_SRC_NODES: int = 10
    NUM_DST_NODES: int = 12

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig()

    @pytest.fixture
    def mapper(self, mapper_init, device):
        return TransformerForwardMapper(**asdict(mapper_init)).to(device)

    @pytest.fixture
    def pair_tensor(self, mapper_init, device):
        return (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )

    def test_custom_attn_channels(self, mapper_init, pair_tensor, device):
        config = asdict(mapper_init)
        config["attn_channels"] = 96

        mapper = TransformerForwardMapper(**config).to(device)

        assert mapper.proc.attention.attn_channels == 96
        assert mapper.proc.attention.projection.in_features == 96
        assert mapper.proc.attention.projection.out_features == mapper_init.hidden_dim

        batch_size = 1
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]
        _, x_dst = mapper.forward(pair_tensor, batch_size, shard_shapes)
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim])

    def test_forward_backward(self, mapper_init, mapper, pair_tensor):
        batch_size = 1
        shard_shapes = [list(pair_tensor[0].shape)], [list(pair_tensor[1].shape)]

        x_src, x_dst = mapper.forward(pair_tensor, batch_size, shard_shapes)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.hidden_dim])

        target = torch.rand(self.NUM_DST_NODES, mapper_init.hidden_dim, device=x_dst.device)
        loss = torch.nn.MSELoss()(x_dst, target)
        loss.backward()

        assert mapper.emb_nodes_src.weight.grad is not None
        assert mapper.emb_nodes_dst.weight.grad is not None
        assert mapper.proc.attention.lin_q.weight.grad is not None
        assert mapper.proc.attention.lin_k.weight.grad is not None
        assert mapper.proc.attention.lin_v.weight.grad is not None
        assert mapper.proc.attention.projection.weight.grad is not None


class TestTransformerBackwardMapper:
    NUM_SRC_NODES: int = 10
    NUM_DST_NODES: int = 12
    OUT_CHANNELS_DST: int = 5

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig()

    @pytest.fixture
    def mapper(self, mapper_init, device):
        return TransformerBackwardMapper(
            **asdict(mapper_init),
            out_channels_dst=self.OUT_CHANNELS_DST,
        ).to(device)

    def test_custom_attn_channels(self, mapper_init, device):
        config = asdict(mapper_init)
        config["attn_channels"] = 96

        mapper = TransformerBackwardMapper(
            **config,
            out_channels_dst=self.OUT_CHANNELS_DST,
        ).to(device)

        assert mapper.proc.attention.attn_channels == 96
        assert mapper.proc.attention.projection.in_features == 96
        assert mapper.proc.attention.projection.out_features == mapper_init.hidden_dim

    def test_forward_backward(self, mapper_init, mapper, device):
        batch_size = 1
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.hidden_dim, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )
        shard_shapes = [[self.NUM_SRC_NODES, mapper_init.in_channels_src]], [
            [self.NUM_DST_NODES, mapper_init.in_channels_dst]
        ]

        out = mapper.forward(x, batch_size, shard_shapes)
        assert out.shape == torch.Size([self.NUM_DST_NODES, self.OUT_CHANNELS_DST])

        target = torch.rand(self.NUM_DST_NODES, self.OUT_CHANNELS_DST, device=out.device)
        loss = torch.nn.MSELoss()(out, target)
        loss.backward()

        assert mapper.emb_nodes_dst.weight.grad is not None
        assert mapper.proc.attention.lin_q.weight.grad is not None
        assert mapper.proc.attention.lin_k.weight.grad is not None
        assert mapper.proc.attention.lin_v.weight.grad is not None
        assert mapper.proc.attention.projection.weight.grad is not None
        assert mapper.node_data_extractor[1].weight.grad is not None
