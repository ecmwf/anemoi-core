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

import ast
import inspect
import textwrap
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.mapper import BaseMapper
from anemoi.models.layers.mapper import GNNForwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.mapper import TransformerForwardMapper
from anemoi.models.layers.processor import BaseProcessor
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.layers.processor import GraphTransformerProcessor
from anemoi.models.layers.processor import PointWiseMLPProcessor
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.sharded_dropout import refuse_dropout_when_sharded
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
    x = (torch.randn(NODES, 3), torch.randn(NODES, 4))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.randn(edge_index.shape[1], EDGE_DIM)
    return mapper(
        x,
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edge_index,
        model_comm_group=model_comm_group,
    )


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


def test_a_positional_model_comm_group_is_found() -> None:
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        transformer(dropout_p=0.1)(
            torch.randn(NODES, CHANNELS), 1, GraphShardInfo(nodes=[NODES]), None, None, sharded_group()
        )


def test_the_check_refuses_only_sharded_training() -> None:
    processor = transformer(dropout_p=0.1)
    with pytest.raises(ValueError, match="Dropout is not supported when the model is sharded"):
        refuse_dropout_when_sharded(processor.train(), sharded_group())
    # Dropout does nothing in eval mode, so sharded inference is not refused.
    refuse_dropout_when_sharded(processor.eval(), sharded_group())
    refuse_dropout_when_sharded(processor.train(), None)
    refuse_dropout_when_sharded(transformer(dropout_p=0.0).train(), sharded_group())


# --- every processor and mapper reaches the check -------------------------------------------


def _is_super_forward(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "forward"
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "super"
    )


def _calls_the_check(call: ast.Call) -> bool:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    return name == "refuse_dropout_when_sharded"


def forward_reaches_the_check(cls: type) -> bool:
    """Follow ``forward`` and its ``super().forward`` calls until one calls the check."""
    for owner in (c for c in cls.__mro__ if "forward" in c.__dict__):
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner.__dict__["forward"])))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if any(_calls_the_check(call) for call in calls):
            return True
        if not any(_is_super_forward(call) for call in calls):
            return False
    return False


def _concrete_subclasses(base: type) -> list[type]:
    found, stack = [], [base]
    while stack:
        for subclass in stack.pop().__subclasses__():
            stack.append(subclass)
            if subclass.__module__.startswith("anemoi.models") and not inspect.isabstract(subclass):
                found.append(subclass)
    return sorted(set(found), key=lambda c: c.__qualname__)


PROCESSORS_AND_MAPPERS = _concrete_subclasses(BaseProcessor) + _concrete_subclasses(BaseMapper)


def test_the_sweep_finds_the_processors_and_mappers() -> None:
    names = {cls.__name__ for cls in PROCESSORS_AND_MAPPERS}
    assert {"TransformerProcessor", "GNNProcessor", "TransformerForwardMapper", "GNNBackwardMapper"} <= names


@pytest.mark.parametrize("cls", PROCESSORS_AND_MAPPERS, ids=lambda cls: cls.__name__)
def test_every_processor_and_mapper_forward_reaches_the_check(cls: type) -> None:
    assert forward_reaches_the_check(cls), (
        f"{cls.__name__}.forward never calls refuse_dropout_when_sharded (directly or through "
        "super().forward). Call it first in forward, so sharded training with dropout is refused."
    )


def test_the_sweep_flags_a_forward_that_skips_the_check() -> None:
    class SkipsTheCheck(TransformerProcessor):
        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
            del args, kwargs
            return x

    assert not forward_reaches_the_check(SkipsTheCheck)
