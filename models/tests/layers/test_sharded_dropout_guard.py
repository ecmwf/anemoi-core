# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Processors and mappers refuse dropout when the model is sharded across GPUs.

Ranks in a model group share a random seed, so dropout on sharded data would draw
correlated masks. The refusal has to cover every processor and mapper, whether the
dropout comes from ``dropout_p`` or from a module added later (a LoRA adapter's
dropout, for instance), and whatever a subclass's ``forward`` does.
"""

import pickle
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import GNNForwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.processor import PointWiseMLPProcessor
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels

CHANNELS, EDGE_DIM, NODES = 16, 3, 4


def _kernels() -> object:
    return load_layer_kernels(instance=False)


def transformer(dropout_p: float = 0.0) -> nn.Module:
    return TransformerProcessor(
        num_layers=2,
        num_channels=CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=_kernels(),
    )


def pointwise(dropout_p: float = 0.0) -> nn.Module:
    return PointWiseMLPProcessor(
        num_layers=2,
        num_channels=CHANNELS,
        num_chunks=1,
        mlp_hidden_ratio=2,
        dropout_p=dropout_p,
        cpu_offload=False,
        layer_kernels=_kernels(),
    )


def gnn() -> nn.Module:
    return GNNProcessor(
        num_channels=CHANNELS,
        num_layers=2,
        num_chunks=1,
        mlp_extra_layers=0,
        cpu_offload=False,
        layer_kernels=_kernels(),
        edge_dim=EDGE_DIM,
    )


def graph_transformer() -> nn.Module:
    return GraphTransformerProcessor(
        num_layers=2,
        num_channels=CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=_kernels(),
        graph_attention_backend="pyg",
        edge_dim=EDGE_DIM,
    )


def with_injected_dropout(processor: nn.Module) -> nn.Module:
    """Put a dropout in front of the first Linear, the way a LoRA adapter's dropout sits."""
    for parent in processor.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                setattr(parent, name, nn.Sequential(nn.Dropout(0.5), child))
                return processor
    msg = "no Linear layer to put a dropout in front of"
    raise AssertionError(msg)


def forward(processor: nn.Module, model_comm_group: object) -> torch.Tensor:
    x = torch.randn(NODES, CHANNELS)
    shard_info = GraphShardInfo(nodes=[NODES])
    if isinstance(processor, GNNProcessor | GraphTransformerProcessor):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = torch.randn(edge_index.shape[1], EDGE_DIM)
        return processor(x, 1, shard_info, edge_attr, edge_index, model_comm_group=model_comm_group)
    return processor(x, 1, shard_info, model_comm_group=model_comm_group)


def sharded_group() -> MagicMock:
    group = MagicMock()
    group.size.return_value = 2
    return group


@pytest.mark.parametrize("build", [transformer, pointwise], ids=["transformer", "pointwise"])
def test_dropout_p_is_refused_when_sharded(build) -> None:
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        forward(build(dropout_p=0.1), sharded_group())


@pytest.mark.parametrize(
    "build",
    [transformer, pointwise, gnn, graph_transformer],
    ids=["transformer", "pointwise", "gnn", "graph_transformer"],
)
def test_added_dropout_module_is_refused_when_sharded(build) -> None:
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        forward(with_injected_dropout(build()), sharded_group())


@pytest.mark.parametrize("build", [transformer, pointwise], ids=["transformer", "pointwise"])
def test_dropout_is_allowed_without_sharding(build) -> None:
    output = forward(with_injected_dropout(build(dropout_p=0.1)), model_comm_group=None)
    assert output.shape == (NODES, CHANNELS)


def transformer_mapper(dropout_p: float = 0.0) -> nn.Module:
    return TransformerForwardMapper(
        in_channels_src=3,
        in_channels_dst=4,
        num_channels=CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        qk_norm=True,
        dropout_p=dropout_p,
        attention_implementation="scaled_dot_product_attention",
        cpu_offload=False,
        layer_kernels=_kernels(),
    )


def graph_transformer_mapper() -> nn.Module:
    return GraphTransformerForwardMapper(
        in_channels_src=3,
        in_channels_dst=4,
        num_channels=CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=_kernels(),
        graph_attention_backend="pyg",
        edge_dim=EDGE_DIM,
    )


def gnn_mapper() -> nn.Module:
    return GNNForwardMapper(
        in_channels_src=3,
        in_channels_dst=4,
        num_channels=CHANNELS,
        num_chunks=1,
        mlp_extra_layers=0,
        edge_dim=EDGE_DIM,
        cpu_offload=False,
        layer_kernels=_kernels(),
    )


def call_mapper(mapper: nn.Module, model_comm_group: object) -> object:
    # The refusal happens before forward runs, so the inputs only need to bind.
    x = (torch.randn(NODES, 3), torch.randn(NODES, 4))
    return mapper(x, batch_size=1, shard_info=BipartiteGraphShardInfo(), model_comm_group=model_comm_group)


def test_mapper_dropout_p_is_refused_when_sharded() -> None:
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        call_mapper(transformer_mapper(dropout_p=0.1), sharded_group())


@pytest.mark.parametrize(
    "build",
    [transformer_mapper, graph_transformer_mapper, gnn_mapper],
    ids=["transformer", "graph_transformer", "gnn"],
)
def test_mapper_added_dropout_module_is_refused_when_sharded(build) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        call_mapper(with_injected_dropout(build()), sharded_group())


class ProcessorThatSkipsEverything(TransformerProcessor):
    """A subclass whose forward neither checks nor calls anything: still refused."""

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        del args, kwargs
        return x


def test_a_subclass_cannot_skip_the_refusal() -> None:
    processor = ProcessorThatSkipsEverything(
        num_layers=2,
        num_channels=CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        dropout_p=0.1,
        attention_implementation="scaled_dot_product_attention",
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=_kernels(),
    )
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        processor(torch.randn(NODES, CHANNELS), 1, GraphShardInfo(nodes=[NODES]), model_comm_group=sharded_group())


def test_dropout_is_allowed_in_eval_mode_when_sharded() -> None:
    # Dropout does nothing in eval mode, so sharded inference of a model trained with
    # dropout must not be refused.
    processor = ProcessorThatSkipsEverything(
        num_layers=2,
        num_channels=CHANNELS,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        dropout_p=0.1,
        attention_implementation="scaled_dot_product_attention",
        qk_norm=True,
        cpu_offload=False,
        layer_kernels=_kernels(),
    ).eval()
    x = torch.randn(NODES, CHANNELS)
    assert processor(x, 1, GraphShardInfo(nodes=[NODES]), model_comm_group=sharded_group()) is x


def test_a_positional_model_comm_group_is_found() -> None:
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        transformer(dropout_p=0.1)(
            torch.randn(NODES, CHANNELS), 1, GraphShardInfo(nodes=[NODES]), None, None, sharded_group()
        )


def test_pickled_modules_carry_no_reference_to_the_refusal() -> None:
    # Inference checkpoints pickle whole modules: nothing about the refusal may end up in
    # the pickle, or renaming it would break every checkpoint written before.
    processor = transformer(dropout_p=0.1)
    data = pickle.dumps(processor)
    assert b"refuse" not in data
    assert b"Guard" not in data
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        forward(pickle.loads(data), sharded_group())
