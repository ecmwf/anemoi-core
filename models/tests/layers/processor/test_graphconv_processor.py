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
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.block import GraphConvProcessorBlock
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.graph_provider import BaseGraphProvider
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict
from tests.distributed._distributed_runner import _run_distributed_test


@dataclass
class GNNProcessorInit:
    """Configuration for the local processor fixtures."""

    num_channels: int = 128
    num_layers: int = 2
    num_chunks: int = 2
    mlp_extra_layers: int = 0
    cpu_offload: bool = False
    gradient_checkpointing: bool = True
    layer_kernels: DotDict = field(default_factory=lambda: load_layer_kernels(instance=False))


class TestGNNProcessor:
    """Test the GNNProcessor class."""

    NUM_NODES: int = 100
    NUM_EDGES: int = 200

    @pytest.fixture(scope="module")
    def component_graph(self, device: torch.device) -> HeteroData:
        """Random graph and two edge-attribute blocks for local processor checks."""
        graph = HeteroData()
        graph["nodes"].x = torch.rand((self.NUM_NODES, 2), device=device)
        graph[("nodes", "to", "nodes")].edge_index = torch.randint(
            0, self.NUM_NODES, (2, self.NUM_EDGES), device=device
        )
        graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((self.NUM_EDGES, 3), device=device)
        graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((self.NUM_EDGES, 4), device=device)
        return graph

    @pytest.fixture
    def graphconv_init(self):
        return GNNProcessorInit()

    @pytest.fixture
    def graph_provider(self, component_graph: HeteroData, device: torch.device) -> BaseGraphProvider:
        """Provide fixed and trainable edge features on the local test device."""
        provider = create_graph_provider(
            graph=component_graph[("nodes", "to", "nodes")],
            edge_attributes=["edge_attr1", "edge_attr2"],
            src_size=self.NUM_NODES,
            dst_size=self.NUM_NODES,
            trainable_size=8,
        )
        return provider.to(device)

    @pytest.fixture
    def graphconv_processor(self, graphconv_init, graph_provider, device):
        return GNNProcessor(**asdict(graphconv_init), edge_dim=graph_provider.edge_dim).to(device)

    def test_graphconv_processor_init(self, graphconv_processor, graphconv_init, graph_provider):
        assert graphconv_processor.num_chunks == graphconv_init.num_chunks
        assert graphconv_processor.num_channels == graphconv_init.num_channels
        assert graphconv_processor.chunk_size == graphconv_init.num_layers // graphconv_init.num_chunks
        assert isinstance(graph_provider.trainable, TrainableTensor)

    def test_all_blocks(self, graphconv_processor):
        assert all(isinstance(block, GraphConvProcessorBlock) for block in graphconv_processor.proc)

    def test_forward(self, graphconv_processor, graphconv_init, graph_provider):
        batch_size = 1
        x = torch.rand(
            (self.NUM_NODES, graphconv_init.num_channels), device=next(graphconv_processor.parameters()).device
        )
        shape_info = GraphShardInfo(nodes=[self.NUM_NODES], edges=[self.NUM_EDGES * batch_size])

        # Run forward pass of processor
        edge_attr, edge_index, _ = graph_provider.get_edges(batch_size=batch_size)
        output = graphconv_processor.forward(x, batch_size, shape_info, edge_attr, edge_index)
        assert output.shape == (self.NUM_NODES, graphconv_init.num_channels)

        # Generate dummy target and loss function
        loss_fn = torch.nn.MSELoss()
        target = torch.rand((self.NUM_NODES, graphconv_init.num_channels), device=output.device)
        loss = loss_fn(output, target)

        # Check loss
        assert loss.item() >= 0

        # Backward pass
        loss.backward()

        # Check gradients of trainable tensor
        assert graph_provider.trainable.trainable.grad.shape == (
            self.NUM_EDGES,
            8,
        )

        # Check gradients of processor
        for param in graphconv_processor.parameters():
            assert param.grad is not None, f"param.grad is None for {param}"
            assert (
                param.grad.shape == param.shape
            ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"

    def test_unsorted_edge_flag_reaches_sharding(
        self, graphconv_processor, graphconv_init, graph_provider, monkeypatch
    ):
        batch_size = 1
        x = torch.rand(
            (self.NUM_NODES, graphconv_init.num_channels), device=next(graphconv_processor.parameters()).device
        )
        shape_info = GraphShardInfo(nodes=[self.NUM_NODES], edges=None)
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

        monkeypatch.setattr("anemoi.models.layers.processor.shard_edges_1hop", fake_shard_edges_1hop)

        output = graphconv_processor.forward(
            x,
            batch_size,
            shape_info,
            edge_attr,
            edge_index,
            edges_are_dst_sorted=False,
        )

        assert called["edges_are_dst_sorted"] is False
        assert output.shape == (self.NUM_NODES, graphconv_init.num_channels)


_SHARDED_EDGE_DIM = 3


@dataclass
class _ProcessorGraph:
    """Controlled node/edge inputs and their ownership, without objective data."""

    node_features: torch.Tensor
    edge_attr: torch.Tensor
    edge_index: torch.Tensor
    node_splits: list[int]
    edge_splits: list[int]


def _build_processor_graph(world_size: int, num_channels: int, device: torch.device) -> _ProcessorGraph:
    """Build full processor inputs for uneven ownership with remote and parallel edges.

    Node features, edge attributes and edge indices are allocated on ``device``.
    The final node is isolated. Returned sizes describe intended ownership;
    this function does not shard the graph or communicate.
    """
    node_splits = [3] + [2] * (world_size - 1)
    starts = [sum(node_splits[:rank]) for rank in range(world_size)]
    num_nodes = sum(node_splits)
    edges = []
    for rank in range(world_size):
        dst = starts[rank]
        remote_src = starts[(rank + 1) % world_size]
        edges.append((remote_src, dst))
        if rank == 0:
            edges.append((remote_src, dst))  # duplicate edge, with a different attribute row
        edges.append((dst, dst))  # local edge
        # Omit the final rank's next-vertex edge to keep the final node isolated.
        if rank < world_size - 1:
            edges.append((dst, dst + 1))
    edge_splits = [
        sum(starts[rank] <= dst < starts[rank] + node_splits[rank] for _, dst in edges) for rank in range(world_size)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()
    # Independently varying channels avoid rows differing only by an offset
    # before the block's LayerNorm operations.
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(num_nodes, num_channels, generator=generator).to(device)
    edge_attr = torch.arange(len(edges) * _SHARDED_EDGE_DIM, dtype=torch.float32, device=device)
    edge_attr = edge_attr.reshape(-1, _SHARDED_EDGE_DIM) / 13 - 0.5
    assert max(node_splits) - min(node_splits) == 1, "Processor fixture must have uneven node shards"
    assert (edge_index[1].diff() >= 0).all(), "Processor edges must be destination-sorted"
    assert torch.equal(edge_index[:, 0], edge_index[:, 1]), "First destination must have parallel edges"
    assert not torch.equal(edge_attr[0], edge_attr[1]), "Parallel edges must have distinct attributes"
    assert not (edge_index == num_nodes - 1).any(), "Final node must be isolated"
    for rank, dst_start in enumerate(starts):
        edge_start = sum(edge_splits[:rank])
        assert edge_index[:, edge_start].tolist() == [
            starts[(rank + 1) % world_size],
            dst_start,
        ], f"Missing remote source entering rank {rank}"
        local_dst = edge_index[1, edge_start : edge_start + edge_splits[rank]]
        assert (
            (local_dst >= starts[rank]) & (local_dst < starts[rank] + node_splits[rank])
        ).all(), f"Edge split does not match destination ownership on rank {rank}"

    return _ProcessorGraph(
        node_features=x,
        edge_attr=edge_attr,
        edge_index=edge_index,
        node_splits=node_splits,
        edge_splits=edge_splits,
    )


def _test_gnn_processor_matches_unsharded_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    pre_sharded: bool,
) -> None:
    """Compare owned outputs and gradients against the same global objective."""
    torch.set_num_threads(1)
    graph = _build_processor_graph(world_size, 8, device)
    x, edge_attr, edge_index = graph.node_features, graph.edge_attr, graph.edge_index
    node_splits, edge_splits = graph.node_splits, graph.edge_splits
    target = torch.linspace(-0.5, 0.6, x.numel(), device=device).reshape_as(x)
    weights = (torch.arange(x.numel(), device=device).reshape_as(x) % 7 + 1).float() / 7
    start = sum(node_splits[:rank])
    stop = start + node_splits[rank]

    torch.manual_seed(17)
    reference = GNNProcessor(
        num_channels=8,
        num_layers=2,
        num_chunks=1,
        mlp_extra_layers=0,
        edge_dim=edge_attr.size(1),
        gradient_checkpointing=False,
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device)
    subject = deepcopy(reference)
    reference_params = dict(reference.named_parameters())
    subject_params = dict(subject.named_parameters())

    x_full = x.clone().requires_grad_(True)
    edges_full = edge_attr.clone().requires_grad_(True)
    y_full = reference(x_full, 1, GraphShardInfo(nodes=[x.size(0)], edges=[edge_index.size(1)]), edges_full, edge_index)
    loss_terms = weights * (y_full - target).square() / weights.numel()
    global_loss = loss_terms.sum()
    remote_loss = loss_terms[:start].sum() + loss_terms[stop:].sum()
    remote_gradient = torch.autograd.grad(remote_loss, x_full, retain_graph=True)[0]
    global_loss.backward()
    gradient_tolerances = {"rtol": 1e-4, "atol": 1e-7}
    assert x_full.grad is not None, "Missing reference node-input gradient"
    allowed_error = gradient_tolerances["atol"] + gradient_tolerances["rtol"] * x_full.grad[start:stop].abs()
    assert (
        remote_gradient[start:stop].abs() > 10 * allowed_error
    ).any(), f"Remote node-gradient contributions are too small to test communication on rank {rank}"

    x_local = x[start:stop].clone().requires_grad_(True)
    edge_start = sum(edge_splits[:rank])
    if pre_sharded:
        input_edges = edge_index[:, edge_start : edge_start + edge_splits[rank]].contiguous()
        edge_features = edge_attr[edge_start : edge_start + edge_splits[rank]].clone().requires_grad_(True)
    else:
        input_edges = edge_index
        edge_features = edge_attr.clone().requires_grad_(True)

    shard_info = GraphShardInfo(nodes=node_splits, edges=edge_splits if pre_sharded else None)
    y_local = subject(x_local, 1, shard_info, edge_features, input_edges, model_comm_group=group)
    local_loss = (weights[start:stop] * (y_local - target[start:stop]).square()).sum() / weights.numel()
    local_loss.backward()

    tolerances = {"rtol": 1e-4, "atol": 1e-5}
    # Callable messages preserve numerical details while adding rank/parameter context.
    torch.testing.assert_close(
        y_local, y_full[start:stop], **tolerances, msg=lambda details: f"output on rank {rank}\n{details}"
    )
    combined_loss = local_loss.detach().clone()
    dist.all_reduce(combined_loss, group=group)
    torch.testing.assert_close(
        combined_loss, global_loss.detach(), **tolerances, msg=lambda details: f"global loss on rank {rank}\n{details}"
    )

    assert x_local.grad is not None, f"Missing distributed node-input gradient on rank {rank}"
    torch.testing.assert_close(
        x_local.grad,
        x_full.grad[start:stop],
        **gradient_tolerances,
        msg=lambda details: f"node-input gradient on rank {rank}\n{details}",
    )
    assert edge_features.grad is not None, f"Missing distributed edge-input gradient on rank {rank}"
    assert edges_full.grad is not None, "Missing reference edge-input gradient"
    if pre_sharded:
        # Edge attributes were already sliced: each rank owns only its edge-gradient rows.
        expected_edge_grad = edges_full.grad[edge_start : edge_start + edge_splits[rank]]
    else:
        # Sharding gathers the full input-edge gradient in backward on each rank.
        expected_edge_grad = edges_full.grad
    torch.testing.assert_close(
        edge_features.grad,
        expected_edge_grad,
        **gradient_tolerances,
        msg=lambda details: f"edge-input gradient on rank {rank}\n{details}",
    )

    # Every layer operates on local edges; its ordinary parameter gradients are partial.
    for name, param in subject_params.items():
        assert param.grad is not None, f"Missing distributed gradient of {name}"
        reference_grad = reference_params[name].grad
        assert reference_grad is not None, f"Missing reference gradient of {name}"
        grad = param.grad.detach().clone()
        dist.all_reduce(grad, group=group)
        torch.testing.assert_close(
            grad,
            reference_grad,
            **tolerances,
            msg=lambda details, name=name: f"gradient of {name} on rank {rank}\n{details}",
        )


@pytest.mark.distributed
@pytest.mark.parametrize("pre_sharded", [False, True], ids=["full-edges", "pre-sharded-edges"])
def test_gnn_processor_matches_unsharded(
    distributed_backend: str,
    distributed_world_size: int,
    pre_sharded: bool,
) -> None:
    """Compare uneven processor shards with the whole-graph objective and derivatives."""
    _run_distributed_test(
        _test_gnn_processor_matches_unsharded_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        pre_sharded=pre_sharded,
    )
