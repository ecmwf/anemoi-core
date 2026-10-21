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
from anemoi.models.layers.mapper import GNNBackwardMapper
from anemoi.models.layers.mapper import GNNBaseMapper
from anemoi.models.layers.mapper import GNNForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict
from tests.distributed._distributed_runner import _run_distributed_test
from tests.distributed._graph_fixtures import build_mapper_graph
from tests.distributed._graph_fixtures import mapper_loss_data


class ConcreteGNNBaseMapper(GNNBaseMapper):
    """Concrete implementation of GNNBaseMapper for testing."""

    def pre_process(self, x):
        x_src, x_dst = x
        return x_src, x_dst

    def post_process(self, x_dst, **kwargs):
        return x_dst


@dataclass
class MapperConfig:
    in_channels_src: int = 3
    in_channels_dst: int = 4
    num_channels: int = 256
    out_channels_dst: int = 8
    num_chunks: int = 2
    mlp_extra_layers: int = 2
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None
    edge_dim: int = None  # Will be set from graph_provider

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


class TestGNNBaseMapper:
    """Test the GNNBaseMapper class."""

    NUM_SRC_NODES: int = 200
    NUM_DST_NODES: int = 178
    NUM_EDGES: int = 300

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
        return ConcreteGNNBaseMapper(**config).to(device)

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
        assert isinstance(mapper, GNNBaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.in_channels_dst == mapper_init.in_channels_dst
        assert mapper.hidden_dim == mapper_init.num_channels
        assert mapper.out_channels_dst == mapper_init.out_channels_dst
        assert mapper.layer_factory is not None

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
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


class TestGNNForwardMapper(TestGNNBaseMapper):
    """Test the GNNForwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        del config["out_channels_dst"]  # Not needed for forward mapper
        return GNNForwardMapper(**config).to(device)

    def test_initialization(self, mapper, mapper_init):
        assert isinstance(mapper, GNNBaseMapper)
        assert mapper.in_channels_src == mapper_init.in_channels_src
        assert mapper.in_channels_dst == mapper_init.in_channels_dst
        assert mapper.hidden_dim == mapper_init.num_channels
        # Forward mapper doesn't have out_channels_dst
        assert mapper.layer_factory is not None

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
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES * batch_size]
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        x_src, x_dst = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.num_channels])
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

    def test_unsorted_edge_flag_reaches_sharding(self, mapper_init, mapper, pair_tensor, graph_provider, monkeypatch):
        x = pair_tensor
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=None)
        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size, shard_edges=False)
        called = {}

        def fake_shard_edges_1hop(
            edge_attr,
            edge_index,
            src_size,
            dst_size,
            model_comm_group,
            edges_are_dst_sorted=True,
        ):
            called["edges_are_dst_sorted"] = edges_are_dst_sorted
            return edge_attr, edge_index, None

        monkeypatch.setattr("anemoi.models.layers.mapper.shard_edges_1hop", fake_shard_edges_1hop)

        x_src, x_dst = mapper.forward(
            x,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            edges_are_dst_sorted=False,
        )

        assert called["edges_are_dst_sorted"] is False
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.num_channels])
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.num_channels])


class TestGNNBackwardMapper(TestGNNBaseMapper):
    """Test the GNNBackwardMapper class."""

    @pytest.fixture
    def mapper(self, mapper_init, graph_provider, device):
        config = asdict(mapper_init)
        config["edge_dim"] = graph_provider.edge_dim
        return GNNBackwardMapper(**config).to(device)

    def test_pre_process(self, mapper, mapper_init, pair_tensor):
        x = pair_tensor

        x_src, x_dst = mapper.pre_process(x)
        assert x_src.shape == torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src]), (
            f"x_src.shape ({x_src.shape}) != torch.Size"
            f"([self.NUM_SRC_NODES, in_channels_src]) ({torch.Size([self.NUM_SRC_NODES, mapper_init.in_channels_src])})"
        )
        assert x_dst.shape == torch.Size([self.NUM_DST_NODES, mapper_init.in_channels_dst]), (
            f"x_dst.shape ({x_dst.shape}) != torch.Size"
            f"([self.NUM_DST_NODES, in_channels_dst]) ({torch.Size([self.NUM_DST_NODES, mapper_init.in_channels_dst])})"
        )

    def test_post_process(self, mapper, mapper_init):
        x_dst = torch.rand(
            self.NUM_DST_NODES,
            mapper_init.num_channels,
            device=next(mapper.parameters()).device,
        )

        result = mapper.post_process(x_dst)
        assert (
            torch.Size([self.NUM_DST_NODES, mapper_init.out_channels_dst]) == result.shape
        ), f"[self.NUM_DST_NODES, out_channels_dst] ({[self.NUM_DST_NODES, mapper_init.out_channels_dst]}) != result.shape ({result.shape})"

    def test_forward_backward(self, mapper_init, mapper, pair_tensor, graph_provider):
        pair_tensor
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(
            src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=[self.NUM_EDGES]
        )

        device = next(mapper.parameters()).device
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.num_channels, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.num_channels, device=device),
        )

        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        result = mapper.forward(x, batch_size, shard_info, edge_attr, edge_index)
        assert result.shape == torch.Size([self.NUM_DST_NODES, mapper_init.out_channels_dst])

        # Dummy loss
        target = torch.rand(self.NUM_DST_NODES, mapper_init.out_channels_dst, device=result.device)
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

    def test_unsorted_edge_flag_reaches_sharding(self, mapper_init, mapper, graph_provider, device, monkeypatch):
        batch_size = 1
        shard_info = BipartiteGraphShardInfo(src_nodes=[self.NUM_SRC_NODES], dst_nodes=[self.NUM_DST_NODES], edges=None)
        x = (
            torch.rand(self.NUM_SRC_NODES, mapper_init.num_channels, device=device),
            torch.rand(self.NUM_DST_NODES, mapper_init.num_channels, device=device),
        )
        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size, shard_edges=False)
        called = {}

        def fake_shard_edges_1hop(
            edge_attr,
            edge_index,
            src_size,
            dst_size,
            model_comm_group,
            edges_are_dst_sorted=True,
        ):
            called["edges_are_dst_sorted"] = edges_are_dst_sorted
            return edge_attr, edge_index, None

        monkeypatch.setattr("anemoi.models.layers.mapper.shard_edges_1hop", fake_shard_edges_1hop)

        result = mapper.forward(
            x,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            edges_are_dst_sorted=False,
        )

        assert called["edges_are_dst_sorted"] is False
        assert result.shape == torch.Size([self.NUM_DST_NODES, mapper_init.out_channels_dst])


def _test_gnn_mapper_matches_unsharded_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    mapper_kind: str,
    keep_x_dst_sharded: bool,
    source_is_sharded: bool,
    gradient_checkpointing: bool,
) -> None:
    """Compare mapper outputs and the derivatives of one globally normalized loss."""
    torch.set_num_threads(1)
    edge_attr, edge_index, src_splits, dst_splits = build_mapper_graph(world_size, device)
    src_start, dst_start = sum(src_splits[:rank]), sum(dst_splits[:rank])
    src_stop, dst_stop = src_start + src_splits[rank], dst_start + dst_splits[rank]
    hidden_channels = 8
    is_forward_mapper = mapper_kind == "forward"
    # The forward mapper embeds raw inputs; the backward mapper consumes latent-width inputs.
    src_channels, dst_channels = (3, 4) if is_forward_mapper else (hidden_channels, hidden_channels)
    # Vary channels independently so latent LayerNorm inputs do not approach constant rows.
    generator = torch.Generator().manual_seed(11)
    x_src = torch.randn(sum(src_splits), src_channels, generator=generator).to(device)
    x_dst = torch.randn(sum(dst_splits), dst_channels, generator=generator).to(device)
    torch.manual_seed(23)
    mapper_cls = GNNForwardMapper if is_forward_mapper else GNNBackwardMapper
    reference = mapper_cls(
        in_channels_src=src_channels,
        in_channels_dst=dst_channels,
        num_channels=hidden_channels,
        out_channels_dst=None if is_forward_mapper else 2,
        num_chunks=1,
        mlp_extra_layers=0,
        edge_dim=edge_attr.size(1),
        gradient_checkpointing=False,
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device)
    subject = deepcopy(reference)
    # Keep the reference uncheckpointed to also verify backward recomputation.
    subject.gradient_checkpointing = gradient_checkpointing
    reference_params = dict(reference.named_parameters())
    subject_params = dict(subject.named_parameters())
    # Reference execution uses all vertices and edges without communication.
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
    reference_dst_output = reference_result[1] if is_forward_mapper else reference_result
    dst_target, dst_weights = mapper_loss_data(reference_dst_output)
    dst_loss_terms = dst_weights * (reference_dst_output - dst_target).square() / dst_weights.numel()
    global_loss = dst_loss_terms.sum()
    if is_forward_mapper:
        reference_src_output = reference_result[0]
        src_target, src_weights = mapper_loss_data(reference_src_output)
        global_loss = (
            global_loss + (src_weights * (reference_src_output - src_target).square()).sum() / src_weights.numel()
        )

    # Isolate contributions from other ranks' destinations. Losing remote-source
    # gradient reduction must change this rank's input gradient beyond our tolerance.
    remote_dst_loss = dst_loss_terms[:dst_start].sum() + dst_loss_terms[dst_stop:].sum()
    remote_src_gradient = torch.autograd.grad(remote_dst_loss, reference_src_input, retain_graph=True)[0]
    tolerances = {"rtol": 1e-4, "atol": 1e-5}
    # Activation-sized absolute tolerances could hide small input gradients.
    gradient_tolerances = {"rtol": 1e-4, "atol": 1e-7}
    global_loss.backward()
    assert reference_src_input.grad is not None, "Missing reference source-input gradient"
    allowed_gradient_error = gradient_tolerances["atol"] + gradient_tolerances["rtol"] * (
        reference_src_input.grad[src_start:src_stop].abs()
    )
    assert (
        remote_src_gradient[src_start:src_stop].abs() > 10 * allowed_gradient_error
    ).any(), f"Remote source-gradient contributions are too small to test communication on rank {rank}"

    # Source input may be replicated; destination inputs are owned rows.
    # Anemoi prepares source and edge shards internally from these declared layouts.
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
    distributed_dst_output = distributed_result[1] if is_forward_mapper else distributed_result
    expected_dst_output = reference_dst_output[dst_start:dst_stop] if keep_x_dst_sharded else reference_dst_output
    # Callable messages add rank/parameter context without replacing numerical diagnostics.
    torch.testing.assert_close(
        distributed_dst_output,
        expected_dst_output,
        **tolerances,
        msg=lambda details: f"destination output on rank {rank}\n{details}",
    )

    # Gather backward splits by owner. Use only owned source/destination terms
    # even when those outputs are gathered, so the rank losses sum to the reference.
    owned_dst_output = distributed_dst_output if keep_x_dst_sharded else distributed_dst_output[dst_start:dst_stop]
    local_loss = (
        dst_weights[dst_start:dst_stop] * (owned_dst_output - dst_target[dst_start:dst_stop]).square()
    ).sum() / dst_weights.numel()
    if is_forward_mapper:
        distributed_src_output = distributed_result[0]
        torch.testing.assert_close(
            distributed_src_output,
            reference_src_output[src_start:src_stop] if source_is_sharded else reference_src_output,
            **tolerances,
            msg=lambda details: f"source output on rank {rank}\n{details}",
        )
        owned_src_output = distributed_src_output if source_is_sharded else distributed_src_output[src_start:src_stop]
        local_loss = (
            local_loss
            + (src_weights[src_start:src_stop] * (owned_src_output - src_target[src_start:src_stop]).square()).sum()
            / src_weights.numel()
        )
    local_loss.backward()
    combined_loss = local_loss.detach().clone()
    dist.all_reduce(combined_loss, group=group)
    torch.testing.assert_close(combined_loss, global_loss.detach(), **tolerances)

    # Compare completed activation gradients and partial parameter contributions.
    assert distributed_src_input.grad is not None, "Missing distributed source-input gradient"
    assert local_dst_input.grad is not None, "Missing distributed destination-input gradient"
    assert reference_dst_input.grad is not None, "Missing reference destination-input gradient"
    torch.testing.assert_close(
        distributed_src_input.grad,
        reference_src_input.grad[src_start:src_stop] if source_is_sharded else reference_src_input.grad,
        **gradient_tolerances,
        msg=lambda details: f"source-input gradient on rank {rank}\n{details}",
    )
    torch.testing.assert_close(
        local_dst_input.grad,
        reference_dst_input.grad[dst_start:dst_stop],
        **gradient_tolerances,
        msg=lambda details: f"destination-input gradient on rank {rank}\n{details}",
    )
    # The mapper shards full edge inputs itself; backward gathers their complete gradient.
    assert full_edge_input.grad is not None, "Missing distributed edge-input gradient"
    assert reference_edge_input.grad is not None, "Missing reference edge-input gradient"
    torch.testing.assert_close(
        full_edge_input.grad,
        reference_edge_input.grad,
        **gradient_tolerances,
        msg=lambda details: f"edge-input gradient on rank {rank}\n{details}",
    )
    for name, param in subject_params.items():
        assert param.grad is not None, f"Missing distributed gradient of {name}"
        reference_grad = reference_params[name].grad
        assert reference_grad is not None, f"Missing reference gradient of {name}"
        # Embeddings, messages, source/destination updates and extraction run on local work.
        grad = param.grad.detach().clone()
        dist.all_reduce(grad, group=group)
        torch.testing.assert_close(
            grad,
            reference_grad,
            **tolerances,
            msg=lambda details, name=name: f"gradient of {name} on rank {rank}\n{details}",
        )


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("mapper_kind", "source_is_sharded", "gradient_checkpointing"),
    [
        pytest.param("forward", True, False, id="forward-mapper-sharded-source"),
        pytest.param("forward", False, False, id="forward-mapper-replicated-source"),
        pytest.param("backward", True, False, id="backward-mapper-sharded-source"),
        pytest.param("forward", False, True, id="forward-mapper-replicated-source-checkpointed"),
    ],
)
@pytest.mark.parametrize("keep_x_dst_sharded", [True, False], ids=["sharded-destinations", "gathered-destinations"])
def test_gnn_mapper_matches_unsharded(
    distributed_backend: str,
    distributed_world_size: int,
    mapper_kind: str,
    keep_x_dst_sharded: bool,
    source_is_sharded: bool,
    gradient_checkpointing: bool,
) -> None:
    """Mapper source layout and destination gathering preserve reference outputs and gradients.

    Both mapper directions run with local or gathered destination outputs. The
    forward mapper preserves replicated or sharded source layout; its objective
    includes the source updates owned by each rank.
    Replicated-source cases also exercise checkpointed distributed execution
    against an uncheckpointed reference.
    """
    _run_distributed_test(
        _test_gnn_mapper_matches_unsharded_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        mapper_kind=mapper_kind,
        keep_x_dst_sharded=keep_x_dst_sharded,
        source_is_sharded=source_is_sharded,
        gradient_checkpointing=gradient_checkpointing,
    )


def test_forward_mapper_replicated_sources_without_group() -> None:
    """An unspecified source partition is equivalent to a single full partition without communication."""
    edge_index = torch.tensor([[0, 2, 1], [0, 0, 1]], dtype=torch.long)
    edge_attr = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 7
    torch.manual_seed(29)
    mapper = GNNForwardMapper(
        in_channels_src=3,
        in_channels_dst=4,
        num_channels=8,
        num_chunks=1,
        mlp_extra_layers=0,
        edge_dim=3,
        gradient_checkpointing=False,
        layer_kernels=load_layer_kernels(instance=False),
    )
    generator = torch.Generator().manual_seed(37)
    src = torch.randn(3, 3, generator=generator)
    dst = torch.randn(2, 4, generator=generator)
    reference_src = src.clone().requires_grad_(True)
    reference_dst = dst.clone().requires_grad_(True)
    actual_src = src.clone().requires_grad_(True)
    actual_dst = dst.clone().requires_grad_(True)
    reference_output = mapper(
        (reference_src, reference_dst),
        1,
        BipartiteGraphShardInfo(src_nodes=[src.size(0)], dst_nodes=[dst.size(0)]),
        edge_attr,
        edge_index,
    )
    actual_output = mapper((actual_src, actual_dst), 1, BipartiteGraphShardInfo(), edge_attr, edge_index)
    for actual, expected in zip(actual_output, reference_output):
        torch.testing.assert_close(actual, expected)
    sum(output.square().sum() for output in reference_output).backward()
    sum(output.square().sum() for output in actual_output).backward()
    torch.testing.assert_close(actual_src.grad, reference_src.grad)
    torch.testing.assert_close(actual_dst.grad, reference_dst.grad)
