# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

import anemoi.models.layers.mapper as mapper_module
import anemoi.models.layers.processor as processor_module
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import GNNBackwardMapper
from anemoi.models.layers.mapper import GNNForwardMapper
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.mapper import TransformerBackwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.processor import PointWiseMLPProcessor
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels


def _layer_kernels():
    return load_layer_kernels(instance=False)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (
            PointWiseMLPProcessor,
            {
                "num_layers": 2,
                "num_channels": 8,
                "num_chunks": 1,
                "mlp_hidden_ratio": 2,
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            TransformerProcessor,
            {
                "num_layers": 2,
                "num_channels": 8,
                "num_chunks": 1,
                "num_heads": 2,
                "mlp_hidden_ratio": 2,
                "attention_implementation": "scaled_dot_product_attention",
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            GNNProcessor,
            {
                "num_layers": 2,
                "num_channels": 8,
                "num_chunks": 1,
                "mlp_extra_layers": 1,
                "edge_dim": 4,
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            GraphTransformerProcessor,
            {
                "num_layers": 2,
                "num_channels": 8,
                "num_chunks": 1,
                "num_heads": 2,
                "mlp_hidden_ratio": 2,
                "edge_dim": 4,
                "graph_attention_backend": "pyg",
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            TransformerForwardMapper,
            {
                "in_channels_src": 4,
                "in_channels_dst": 4,
                "hidden_dim": 8,
                "out_channels_dst": None,
                "num_chunks": 1,
                "num_heads": 2,
                "mlp_hidden_ratio": 2,
                "attention_implementation": "scaled_dot_product_attention",
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            TransformerBackwardMapper,
            {
                "in_channels_src": 8,
                "in_channels_dst": 4,
                "hidden_dim": 8,
                "out_channels_dst": 3,
                "num_chunks": 1,
                "num_heads": 2,
                "mlp_hidden_ratio": 2,
                "attention_implementation": "scaled_dot_product_attention",
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            GNNForwardMapper,
            {
                "in_channels_src": 4,
                "in_channels_dst": 4,
                "hidden_dim": 8,
                "out_channels_dst": None,
                "num_chunks": 1,
                "mlp_extra_layers": 1,
                "edge_dim": 4,
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            GNNBackwardMapper,
            {
                "in_channels_src": 8,
                "in_channels_dst": 4,
                "hidden_dim": 8,
                "out_channels_dst": 3,
                "num_chunks": 1,
                "mlp_extra_layers": 1,
                "edge_dim": 4,
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            GraphTransformerForwardMapper,
            {
                "in_channels_src": 4,
                "in_channels_dst": 4,
                "hidden_dim": 8,
                "num_chunks": 1,
                "num_heads": 2,
                "mlp_hidden_ratio": 2,
                "edge_dim": 4,
                "graph_attention_backend": "pyg",
                "out_channels_dst": None,
                "layer_kernels": _layer_kernels(),
            },
        ),
        (
            GraphTransformerBackwardMapper,
            {
                "in_channels_src": 8,
                "in_channels_dst": 4,
                "hidden_dim": 8,
                "out_channels_dst": 3,
                "num_chunks": 1,
                "num_heads": 2,
                "mlp_hidden_ratio": 2,
                "edge_dim": 4,
                "graph_attention_backend": "pyg",
                "layer_kernels": _layer_kernels(),
            },
        ),
    ],
)
def test_gradient_checkpointing_is_forwarded_to_concrete_layers(cls, kwargs):
    layer = cls(**kwargs, gradient_checkpointing=False)
    assert layer.gradient_checkpointing is False


def test_processor_forward_uses_disabled_checkpoint_flag(monkeypatch):
    calls = []

    def fake_maybe_checkpoint(func, enabled, *args, **kwargs):
        calls.append(enabled)
        return func(*args, **kwargs)

    monkeypatch.setattr(processor_module, "maybe_checkpoint", fake_maybe_checkpoint)

    processor = PointWiseMLPProcessor(
        num_layers=2,
        num_channels=8,
        num_chunks=2,
        mlp_hidden_ratio=2,
        gradient_checkpointing=False,
        layer_kernels=_layer_kernels(),
    )

    x = torch.rand(10, 8)
    shard_info = GraphShardInfo(nodes=[10])
    output = processor(x, batch_size=1, shard_info=shard_info)

    assert output.shape == x.shape
    assert calls == [False, False]


def test_mapper_forward_uses_disabled_checkpoint_flag(monkeypatch):
    calls = []

    def fake_maybe_checkpoint(func, enabled, *args, **kwargs):
        calls.append(enabled)
        return func(*args, **kwargs)

    monkeypatch.setattr(mapper_module, "maybe_checkpoint", fake_maybe_checkpoint)

    mapper = TransformerForwardMapper(
        in_channels_src=4,
        in_channels_dst=4,
        hidden_dim=8,
        out_channels_dst=None,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        attention_implementation="scaled_dot_product_attention",
        gradient_checkpointing=False,
        layer_kernels=_layer_kernels(),
    )

    x = (torch.rand(6, 4), torch.rand(5, 4))
    shard_info = BipartiteGraphShardInfo(src_nodes=[6], dst_nodes=[5])
    x_src, x_dst = mapper(x, batch_size=1, shard_info=shard_info)

    assert x_src.shape == x[0].shape
    assert x_dst.shape == (5, 8)
    assert calls == [False]
