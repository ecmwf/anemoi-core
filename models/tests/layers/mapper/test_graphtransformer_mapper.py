# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict
from tests.distributed._distributed_runner import _run_distributed_test
from tests.distributed._graph_fixtures import build_mapper_graph
from tests.distributed._graph_fixtures import mapper_loss_data


class ConcreteGraphTransformerBaseMapper(GraphTransformerBaseMapper):
    """Concrete implementation of GraphTransformerBaseMapper for testing."""

    def pre_process(self, x):
        x_src, x_dst = x
        return x_src, x_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst


@dataclass
class MapperConfig:
    in_channels_src: int = 5
    in_channels_dst: int = 3
    num_channels: int = 256
    num_chunks: int = 2
    num_heads: int = 16
    mlp_hidden_ratio: int = 7
    attn_channels: int | None = None
    qk_norm: bool = True
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None
    shard_strategy: str = "edges"
    graph_attention_backend: str = "pyg"
    edge_dim: int = None  # Will be set from graph_provider
    edge_pre_mlp: bool = False

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


class TestGraphTransformerBaseMapper:
    """Test the GraphTransformerBaseMapper class."""

    NUM_EDGES: int = 150
    NUM_SRC_NODES: int = 100
    NUM_DST_NODES: int = 200
    OUT_CHANNELS_DST: int = 5

    @pytest.fixture
    def mapper_init(self):
        return MapperConfig()

    @pytest.fixture
    def graph_provider(self, fake_graph, device):
        provider = create_graph_provider(
            graph=fake_graph[("nodes", "to", "nodes")],
            edge_attributes=["edge_attr1", "edge_attr2"],
            src_size=self.NUM_SRC_NODES,
            dst_size=self.NUM_DST_NODES,
            trainable_size=6,
        )
        return provider.to(device)

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return ConcreteGraphTransformerBaseMapper(
            **config,
            out_channels_dst=self.OUT_CHANNELS_DST,
        ).to(device)

    @pytest.fixture
    def pair_tensor(self, mapper_init, device):
        return (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )

    @pytest.fixture(scope="module")
    def fake_graph(self, device) -> HeteroData:
        """Fake graph."""
        graph = HeteroData()
        graph[("nodes", "to", "nodes")].edge_index = torch.concat(
            [
                torch.randint(0, self.NUM_SRC_NODES, (1, self.NUM_EDGES), device=device),
                torch.randint(0, self.NUM_DST_NODES, (1, self.NUM_EDGES), device=device),
            ],
            axis=0,
        )
        graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((self.NUM_EDGES, 1), device=device)
        graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((self.NUM_EDGES, 32), device=device)
        return graph

    def test_initialization(self, mapper, mapper_init):
        assert isinstance(mapper, GraphTransformerBaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.in_channels_dst == mapper_init.in_channels_dst
        assert mapper.hidden_dim == mapper_init.num_channels
        assert mapper.out_channels_dst == self.OUT_CHANNELS_DST
        assert mapper.layer_factory is not None

    def test_pre_process(self, mapper, pair_tensor):
        # Should be a no-op in the base class
        x = pair_tensor

        x_src, x_dst = mapper.pre_process(x)
        assert x_src.shape == torch.Size(
            x[0].shape
        ), f"x_src.shape ({x_src.shape}) != torch.Size(x[0].shape) ({torch.Size(x[0].shape)})"
        assert x_dst.shape == torch.Size(
            x[1].shape
        ), f"x_dst.shape ({x_dst.shape}) != torch.Size(x[1].shape) ({x[1].shape})"

    def test_post_process(self, mapper, pair_tensor):
        # Should be a no-op in the base class
        x_dst = pair_tensor[1]

        result = mapper.post_process(x_dst)
        assert torch.equal(result, x_dst)


class TestGraphTransformerForwardMapper(TestGraphTransformerBaseMapper):
    """Test the GraphTransformerForwardMapper class."""

    OUT_CHANNELS_DST = None

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return GraphTransformerForwardMapper(**config).to(device)

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor

        x_src, x_dst = mapper.pre_process(x)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.num_channels]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, hidden_dim]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.num_channels])})"
        )
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.num_channels]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            "([self.NUM_DST_NODES, hidden_dim]) ({torch.Size([self.NUM_DST_NODES, hidden_dim])})"
        )

    def test_forward_backward(self, mapper_init, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        x_src, x_dst = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.num_channels])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, mapper_init.num_channels, device=x_dst.device)
        loss_fn = nn.MSELoss()

        loss = loss_fn(x_dst, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert graph_provider.trainable.trainable.grad is not None
        assert graph_provider.trainable.trainable.grad.shape == graph_provider.trainable.trainable.shape

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"

    def test_chunking(self, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        mapper.num_chunks = 4
        x_src_c, x_dst_c = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)

        mapper.num_chunks = 1
        x_src, x_dst = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)

        assert torch.allclose(
            x_src, x_src_c, atol=1e-4
        ), f"x_src ({x_src}) != x_src_c ({x_src_c}) when num_chunks is changed"
        assert torch.allclose(
            x_dst, x_dst_c, atol=1e-4
        ), f"x_dst ({x_dst}) != x_dst_c ({x_dst_c}) when num_chunks is changed"

    def test_unsorted_edges_are_sorted_before_partitioning(self, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        mapper.num_chunks = 4
        x_src_sorted, x_dst_sorted = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)

        perm = torch.randperm(edge_index.shape[1], device=edge_index.device)
        x_src_unsorted, x_dst_unsorted = mapper.forward(
            x, batch_size, shard_info, edge_attr[perm], edge_index[:, perm], edges_are_dst_sorted=False
        )

        assert torch.allclose(x_src_sorted, x_src_unsorted, atol=1e-4)
        assert torch.allclose(x_dst_sorted, x_dst_unsorted, atol=1e-4)

    def test_strategy(self, mapper, pair_tensor, graph_provider):
        x = pair_tensor
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        out_heads = mapper.mapper_forward_with_heads_sharding(x, batch_size, shard_info, edge_attr, edge_index)

        out_edges = mapper.mapper_forward_with_edge_sharding(x, batch_size, shard_info, edge_attr, edge_index)

        assert torch.allclose(
            out_heads, out_edges, atol=1e-4
        ), f"out_heads ({out_heads}) != out_edges ({out_edges}) when using different strategies"

    def test_custom_attn_channels(self, mapper_init, graph_provider, pair_tensor, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        config["attn_channels"] = 112

        mapper = GraphTransformerForwardMapper(**config).to(device)

        assert mapper.proc.attn_channels == 112
        assert mapper.proc.projection.in_features == 112
        assert mapper.proc.projection.out_features == mapper_init.num_channels

        batch_size = 1
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )
        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        _, x_dst = mapper.forward(pair_tensor, batch_size, shard_info, edge_attr, edge_index)
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.num_channels])


class TestGraphTransformerBackwardMapper(TestGraphTransformerBaseMapper):
    """Test the GraphTransformerBackwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return GraphTransformerBackwardMapper(
            **config,
            out_channels_dst=self.OUT_CHANNELS_DST,
        ).to(device)

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor

        x_src, x_dst = mapper.pre_process(x)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.num_channels]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, num_channels]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.num_channels])})"
        )
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.num_channels]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            f"([self.NUM_DST_NODES, num_channels]) ({torch.Size([self.NUM_DST_NODES, mapper_init.num_channels])})"
        )

    def test_post_process(self, mapper, mapper_init):
        x_dst = torch.rand(
            self.NUM_DST_NODES,
            mapper_init.num_channels,
            device=next(mapper.parameters()).device,
        )

        result = mapper.post_process(x_dst)
        assert (
            torch.Size([self.NUM_DST_NODES, self.OUT_CHANNELS_DST]) == result.shape
        ), f"[self.NUM_DST_NODES, out_channels_dst] ({[self.NUM_DST_NODES, self.OUT_CHANNELS_DST]}) != result.shape ({result.shape})"

    def test_forward_backward(self, mapper_init, mapper, pair_tensor, graph_provider):
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )
        batch_size = 1

        # Different size for x_dst, as the Backward mapper changes the channels in shape in pre-processor
        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        result = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)
        assert result.shape == torch.Size([self.NUM_DST_NODES, self.OUT_CHANNELS_DST])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, self.OUT_CHANNELS_DST, device=result.device)
        loss_fn = nn.MSELoss()

        loss = loss_fn(result, target)

        # Check loss
        assert loss.item() >= 0

        loss.backward()

        # Check gradients
        assert graph_provider.trainable.trainable.grad is not None
        assert graph_provider.trainable.trainable.grad.shape == graph_provider.trainable.trainable.shape

        for param in mapper.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"

    def test_chunking(self, mapper_init, mapper, pair_tensor, graph_provider):
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )
        batch_size = 1

        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        mapper.num_chunks = 4
        out_c = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)

        mapper.num_chunks = 1
        out = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)

        assert torch.allclose(out, out_c, atol=1e-4), f"out ({out}) != out_c ({out_c}) when num_chunks is changed"

    def test_strategy(self, mapper_init, mapper, pair_tensor, graph_provider):
        shard_info = BipartiteGraphShardInfo(src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES])
        batch_size = 1

        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.in_channels_src, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.in_channels_dst, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)

        out_heads = mapper.mapper_forward_with_heads_sharding(x, batch_size, shard_info, edge_attr, edge_index)

        out_edges = mapper.mapper_forward_with_edge_sharding(x, batch_size, shard_info, edge_attr, edge_index)

        assert torch.allclose(
            out_heads, out_edges, atol=1e-4
        ), f"out_heads ({out_heads}) != out_edges ({out_edges}) when using different strategies"


def _test_graphtransformer_mapper_matches_unsharded_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    mapper_kind: str,
    source_is_sharded: bool,
    keep_x_dst_sharded: bool,
    gradient_checkpointing: bool,
) -> None:
    """Compare distributed layouts and gradients with an unsharded reference.

    Replicated source passthrough uses the full direct source loss on each
    rank for backward, but only owned source terms enter the detached scalar
    used to reconstruct the global loss.
    """
    torch.set_num_threads(1)
    edge_attr, edge_index, src_splits, dst_splits = build_mapper_graph(world_size, device)
    src_start, src_stop = sum(src_splits[:rank]), sum(src_splits[: rank + 1])
    dst_start, dst_stop = sum(dst_splits[:rank]), sum(dst_splits[: rank + 1])
    # Guard the graph properties that make this a communication/parallel-edge regression.
    assert (edge_index[1].diff() >= 0).all(), "Mapper edges must be destination-sorted"
    assert torch.equal(edge_index[:, 0], edge_index[:, 2]), "Missing parallel remote edge"
    assert not torch.equal(edge_attr[0], edge_attr[2]), "Parallel edges need distinct features"
    sources = edge_index[0, (edge_index[1] >= dst_start) & (edge_index[1] < dst_stop)]
    owned = (sources >= src_start) & (sources < src_stop)
    assert (~owned).any(), f"Missing remote source dependency for rank {rank}"
    assert sources[owned].unique().numel() > 1, f"Insufficient owned source variation for rank {rank}"
    is_forward_mapper = mapper_kind == "forward"
    src_channels, dst_channels = (5, 3) if is_forward_mapper else (8, 5)
    generator = torch.Generator().manual_seed(53)
    x_src = torch.randn(sum(src_splits), src_channels, generator=generator).to(device)
    x_dst = torch.randn(sum(dst_splits), dst_channels, generator=generator).to(device)
    torch.manual_seed(59)
    mapper_cls = GraphTransformerForwardMapper if is_forward_mapper else GraphTransformerBackwardMapper
    reference = mapper_cls(
        in_channels_src=src_channels,
        in_channels_dst=dst_channels,
        num_channels=8,
        out_channels_dst=None if is_forward_mapper else 2,
        num_chunks=1,
        num_heads=2,
        mlp_hidden_ratio=2,
        edge_dim=edge_attr.size(1),
        qk_norm=True,
        gradient_checkpointing=False,
        shard_strategy="edges",
        graph_attention_backend="pyg",
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device)
    subject = deepcopy(reference)
    # Check recomputation against an uncheckpointed reference.
    subject.gradient_checkpointing = gradient_checkpointing
    reference_params = dict(reference.named_parameters())
    subject_params = dict(subject.named_parameters())
    reference_src_input = x_src.clone().requires_grad_(True)
    reference_dst_input = x_dst.clone().requires_grad_(True)
    reference_edge_input = edge_attr.clone().requires_grad_(True)
    reference_result = reference(
        (reference_src_input, reference_dst_input),
        1,
        BipartiteGraphShardInfo(src_nodes=[sum(src_splits)], dst_nodes=[sum(dst_splits)]),
        reference_edge_input,
        edge_index,
        keep_x_dst_sharded=True,
    )
    reference_output = reference_result[1] if is_forward_mapper else reference_result
    target, weights = mapper_loss_data(reference_output)
    loss_terms = weights * (reference_output - target).square() / weights.numel()
    global_loss = loss_terms.sum()
    if is_forward_mapper:
        src_target = torch.linspace(-0.4, 0.6, x_src.numel(), device=device).reshape_as(x_src)
        torch.testing.assert_close(reference_result[0], reference_src_input)
        global_loss = global_loss + (reference_result[0] - src_target).square().sum() / x_src.numel()

    # Require detectable remote attention sensitivity, independently of the source passthrough loss.
    remote_loss = loss_terms[:dst_start].sum() + loss_terms[dst_stop:].sum()
    remote_gradient = torch.autograd.grad(remote_loss, reference_src_input, retain_graph=True)[0]
    global_loss.backward()
    tolerances = {"rtol": 1e-4, "atol": 1e-5}
    gradient_tolerances = {"rtol": 1e-4, "atol": 1e-7}
    assert reference_src_input.grad is not None, "Missing reference source-input gradient"
    allowed_error = gradient_tolerances["atol"] + gradient_tolerances["rtol"] * (
        reference_src_input.grad[src_start:src_stop].abs()
    )
    assert (
        remote_gradient[src_start:src_stop].abs() > 10 * allowed_error
    ).any(), f"Remote attention source gradients are too small to test communication on rank {rank}"

    distributed_src_input = (x_src[src_start:src_stop] if source_is_sharded else x_src).clone().requires_grad_(True)
    local_dst_input = x_dst[dst_start:dst_stop].clone().requires_grad_(True)
    full_edge_input = edge_attr.clone().requires_grad_(True)
    distributed_result = subject(
        (distributed_src_input, local_dst_input),
        1,
        BipartiteGraphShardInfo(src_nodes=src_splits if source_is_sharded else None, dst_nodes=dst_splits),
        full_edge_input,
        edge_index,
        model_comm_group=group,
        keep_x_dst_sharded=keep_x_dst_sharded,
    )
    distributed_output = distributed_result[1] if is_forward_mapper else distributed_result
    # Callable messages add rank/parameter context without replacing numerical diagnostics.
    torch.testing.assert_close(
        distributed_output,
        reference_output[dst_start:dst_stop] if keep_x_dst_sharded else reference_output,
        **tolerances,
        msg=lambda details: f"destination output on rank {rank}\n{details}",
    )
    # Gather backward splits by owner: use owned destination terms in either output layout.
    owned_output = distributed_output if keep_x_dst_sharded else distributed_output[dst_start:dst_stop]
    local_loss = (weights[dst_start:dst_stop] * (owned_output - target[dst_start:dst_stop]).square()).sum()
    local_loss = local_loss / weights.numel()
    owned_loss_value = local_loss.detach().clone()
    if is_forward_mapper:
        # Replicated passthrough is a direct path without gradient communication.
        # Use its full source objective on each rank to obtain the full input gradient;
        # only its owned terms enter the scalar loss comparison below.
        expected_source = reference_result[0][src_start:src_stop] if source_is_sharded else reference_result[0]
        distributed_source_target = src_target[src_start:src_stop] if source_is_sharded else src_target
        torch.testing.assert_close(
            distributed_result[0], expected_source, msg=lambda details: f"source output on rank {rank}\n{details}"
        )
        source_terms = (distributed_result[0] - distributed_source_target).square()
        source_loss = source_terms.sum() / x_src.numel()
        local_loss = local_loss + source_loss
        owned_source_loss = (
            source_terms if source_is_sharded else source_terms[src_start:src_stop]
        ).sum() / x_src.numel()
        owned_loss_value = owned_loss_value + owned_source_loss.detach()
    local_loss.backward()
    dist.all_reduce(owned_loss_value, group=group)
    torch.testing.assert_close(
        owned_loss_value,
        global_loss.detach(),
        **tolerances,
        msg=lambda details: f"global loss on rank {rank}\n{details}",
    )

    # Production edge sharding gathers the full edge-input gradient in backward.
    for name, actual, expected, rows in (
        (
            "source input",
            distributed_src_input,
            reference_src_input,
            slice(src_start, src_stop) if source_is_sharded else slice(None),
        ),
        ("destination input", local_dst_input, reference_dst_input, slice(dst_start, dst_stop)),
        ("edge input", full_edge_input, reference_edge_input, slice(None)),
    ):
        assert actual.grad is not None, f"Missing distributed {name} gradient on rank {rank}"
        assert expected.grad is not None, f"Missing reference {name} gradient"
        torch.testing.assert_close(
            actual.grad,
            expected.grad[rows],
            **gradient_tolerances,
            msg=lambda details, name=name: f"{name} gradient on rank {rank}\n{details}",
        )
    for name, param in subject_params.items():
        assert param.grad is not None, f"Missing distributed gradient of {name}"
        reference_gradient = reference_params[name].grad
        assert reference_gradient is not None, f"Missing reference gradient of {name}"
        # Projections and attention/MLP layers operate on rank-local subgraphs.
        gradient = param.grad.detach().clone()
        dist.all_reduce(gradient, group=group)
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            **gradient_tolerances,
            msg=lambda details, name=name: f"gradient of {name} on rank {rank}\n{details}",
        )


@pytest.mark.distributed
@pytest.mark.parametrize("mapper_kind", ["forward", "backward"], ids=["forward-mapper", "backward-mapper"])
@pytest.mark.parametrize("keep_x_dst_sharded", [True, False], ids=["sharded-destinations", "gathered-destinations"])
@pytest.mark.parametrize(
    ("source_is_sharded", "gradient_checkpointing"),
    [
        pytest.param(True, False, id="sharded-source"),
        pytest.param(False, False, id="replicated-source"),
        pytest.param(False, True, id="replicated-source-checkpointed"),
    ],
)
def test_graphtransformer_mapper_matches_unsharded(
    distributed_backend: str,
    distributed_world_size: int,
    mapper_kind: str,
    source_is_sharded: bool,
    keep_x_dst_sharded: bool,
    gradient_checkpointing: bool,
) -> None:
    """PyG edge sharding preserves layouts and gradients, including checkpoint recomputation.

    Gathered destination losses use owned terms. Replicated source passthrough
    requires a full direct source loss on each rank for its complete input
    derivative; only owned source terms enter the summed loss comparison.
    """
    _run_distributed_test(
        _test_graphtransformer_mapper_matches_unsharded_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        mapper_kind=mapper_kind,
        source_is_sharded=source_is_sharded,
        keep_x_dst_sharded=keep_x_dst_sharded,
        gradient_checkpointing=gradient_checkpointing,
    )
