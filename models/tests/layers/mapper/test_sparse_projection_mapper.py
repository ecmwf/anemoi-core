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

from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.mapper import SparseProjectionBackwardMapper
from anemoi.models.layers.mapper import SparseProjectionForwardMapper
from anemoi.models.layers.utils import load_layer_kernels

IN_CHANNELS_SRC = 3
IN_CHANNELS_DST = 2
NUM_CHANNELS = 8
NUM_SRC = 20
NUM_DST = 5
OUT_CHANNELS_DST = 4


def _build_mapper(device, row_normalize: bool = True) -> SparseProjectionForwardMapper:
    return SparseProjectionForwardMapper(
        in_channels_src=IN_CHANNELS_SRC,
        in_channels_dst=IN_CHANNELS_DST,
        num_channels=NUM_CHANNELS,
        edge_dim=1,
        row_normalize=row_normalize,
        layer_kernels=load_layer_kernels(instance=False),
    ).to(device)


@pytest.fixture
def graph(device):
    """Each dst node is connected to 4 src nodes, edges sorted by dst."""
    dst = torch.arange(NUM_DST).repeat_interleave(4)
    src = torch.randperm(NUM_SRC)[: dst.numel()]
    edge_index = torch.stack([src, dst]).to(device)
    edge_attr = torch.rand(edge_index.shape[1], 1, device=device) + 0.1
    return edge_attr, edge_index


@pytest.fixture
def nodes(device):
    return (
        torch.randn(NUM_SRC, IN_CHANNELS_SRC, device=device),
        torch.randn(NUM_DST, IN_CHANNELS_DST, device=device),
    )


def _dense_matrix(edge_attr, edge_index, row_normalize, num_src=NUM_SRC, num_dst=NUM_DST):
    matrix = torch.zeros(num_dst, num_src, device=edge_attr.device)
    matrix.index_put_((edge_index[1], edge_index[0]), edge_attr[:, 0], accumulate=True)
    if row_normalize:
        matrix = matrix / matrix.sum(dim=1, keepdim=True)
    return matrix


def _dense_reference(mapper, x_src, edge_attr, edge_index, row_normalize):
    return mapper.emb_nodes_src(_dense_matrix(edge_attr, edge_index, row_normalize) @ x_src)


@pytest.mark.parametrize("row_normalize", [True, False])
def test_sparse_projection_matches_dense(device, graph, nodes, row_normalize):
    mapper = _build_mapper(device, row_normalize=row_normalize)
    edge_attr, edge_index = graph

    x_src, x_hidden = mapper(
        nodes,
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )

    assert torch.equal(x_src, nodes[0])
    assert x_hidden.shape == (NUM_DST, NUM_CHANNELS)
    expected = _dense_reference(mapper, nodes[0], edge_attr, edge_index, row_normalize)
    torch.testing.assert_close(x_hidden, expected)


def test_sparse_projection_unsorted_edges(device, graph, nodes):
    mapper = _build_mapper(device)
    edge_attr, edge_index = graph
    perm = torch.randperm(edge_index.shape[1], device=device)

    _, x_hidden = mapper(
        nodes,
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr[perm],
        edge_index=edge_index[:, perm],
        edges_are_dst_sorted=False,
    )

    expected = _dense_reference(mapper, nodes[0], edge_attr, edge_index, row_normalize=True)
    torch.testing.assert_close(x_hidden, expected)


def test_sparse_projection_batched_edges(device, graph, nodes):
    """Edges expanded over the batch, as returned by StaticGraphProvider, project each sample independently."""
    mapper = _build_mapper(device)
    edge_attr, edge_index = graph
    batch_size = 2
    edge_inc = torch.tensor([[NUM_SRC], [NUM_DST]], device=device)
    edge_index_batched = torch.cat([edge_index + i * edge_inc for i in range(batch_size)], dim=1)
    edge_index_batched, perm = sort_edge_index_by_dst(edge_index_batched)
    edge_attr_batched = edge_attr.repeat(batch_size, 1)[perm]
    x_src = torch.randn(batch_size * NUM_SRC, IN_CHANNELS_SRC, device=device)
    x_dst = torch.randn(batch_size * NUM_DST, IN_CHANNELS_DST, device=device)

    _, x_hidden = mapper(
        (x_src, x_dst),
        batch_size=batch_size,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr_batched,
        edge_index=edge_index_batched,
    )

    for i in range(batch_size):
        expected = _dense_reference(
            mapper, x_src[i * NUM_SRC : (i + 1) * NUM_SRC], edge_attr, edge_index, row_normalize=True
        )
        torch.testing.assert_close(x_hidden[i * NUM_DST : (i + 1) * NUM_DST], expected)


def test_sparse_projection_backward(device, graph, nodes):
    mapper = _build_mapper(device)
    edge_attr, edge_index = graph
    x_src = nodes[0].clone().requires_grad_(True)

    _, x_hidden = mapper(
        (x_src, nodes[1]),
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )
    x_hidden.sum().backward()

    assert x_src.grad is not None
    assert mapper.emb_nodes_src.weight.grad is not None
    # src nodes without edges receive no gradient
    unconnected = torch.ones(NUM_SRC, dtype=torch.bool, device=device)
    unconnected[edge_index[0]] = False
    assert torch.all(x_src.grad[unconnected] == 0)


def test_sparse_projection_requires_single_edge_attribute():
    with pytest.raises(AssertionError, match="exactly one edge attribute"):
        SparseProjectionForwardMapper(
            in_channels_src=IN_CHANNELS_SRC,
            in_channels_dst=IN_CHANNELS_DST,
            num_channels=NUM_CHANNELS,
            edge_dim=3,
            layer_kernels=load_layer_kernels(instance=False),
        )


# Backward mapper: hidden (NUM_DST nodes of the forward graph) -> data (NUM_SRC nodes)


def _build_backward_mapper(device, row_normalize: bool = True, **kwargs) -> SparseProjectionBackwardMapper:
    return SparseProjectionBackwardMapper(
        in_channels_src=NUM_CHANNELS,
        in_channels_dst=IN_CHANNELS_SRC,
        num_channels=NUM_CHANNELS,
        out_channels_dst=OUT_CHANNELS_DST,
        edge_dim=1,
        row_normalize=row_normalize,
        layer_kernels=load_layer_kernels(instance=False),
        **kwargs,
    ).to(device)


@pytest.fixture
def backward_graph(device):
    """Each data node is connected to 2 hidden nodes, edges sorted by dst."""
    dst = torch.arange(NUM_SRC).repeat_interleave(2)
    src = torch.randint(0, NUM_DST, (dst.numel(),))
    edge_index = torch.stack([src, dst]).to(device)
    edge_attr = torch.rand(edge_index.shape[1], 1, device=device) + 0.1
    return edge_attr, edge_index


@pytest.fixture
def backward_nodes(device):
    return (
        torch.randn(NUM_DST, NUM_CHANNELS, device=device),
        torch.randn(NUM_SRC, IN_CHANNELS_SRC, device=device),
    )


@pytest.mark.parametrize("row_normalize", [True, False])
def test_sparse_projection_backward_matches_dense(device, backward_graph, backward_nodes, row_normalize):
    mapper = _build_backward_mapper(device, row_normalize=row_normalize)
    edge_attr, edge_index = backward_graph

    x_out = mapper(
        backward_nodes,
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )

    assert x_out.shape == (NUM_SRC, OUT_CHANNELS_DST)
    matrix = _dense_matrix(edge_attr, edge_index, row_normalize, num_src=NUM_DST, num_dst=NUM_SRC)
    expected = matrix @ mapper.node_data_extractor(backward_nodes[0])
    torch.testing.assert_close(x_out, expected)


def test_sparse_projection_backward_batched_edges(device, backward_graph):
    mapper = _build_backward_mapper(device)
    edge_attr, edge_index = backward_graph
    batch_size = 2
    edge_inc = torch.tensor([[NUM_DST], [NUM_SRC]], device=device)
    edge_index_batched = torch.cat([edge_index + i * edge_inc for i in range(batch_size)], dim=1)
    edge_index_batched, perm = sort_edge_index_by_dst(edge_index_batched)
    edge_attr_batched = edge_attr.repeat(batch_size, 1)[perm]
    x_hidden = torch.randn(batch_size * NUM_DST, NUM_CHANNELS, device=device)
    x_data = torch.randn(batch_size * NUM_SRC, IN_CHANNELS_SRC, device=device)

    x_out = mapper(
        (x_hidden, x_data),
        batch_size=batch_size,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr_batched,
        edge_index=edge_index_batched,
    )

    matrix = _dense_matrix(edge_attr, edge_index, row_normalize=True, num_src=NUM_DST, num_dst=NUM_SRC)
    for i in range(batch_size):
        expected = matrix @ mapper.node_data_extractor(x_hidden[i * NUM_DST : (i + 1) * NUM_DST])
        torch.testing.assert_close(x_out[i * NUM_SRC : (i + 1) * NUM_SRC], expected)


def test_sparse_projection_backward_gradients(device, backward_graph, backward_nodes):
    mapper = _build_backward_mapper(device)
    edge_attr, edge_index = backward_graph
    x_hidden = backward_nodes[0].clone().requires_grad_(True)

    x_out = mapper(
        (x_hidden, backward_nodes[1]),
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )
    x_out.sum().backward()

    assert x_hidden.grad is not None
    for param in mapper.parameters():
        assert param.grad is not None


def test_sparse_projection_backward_zero_init(device, backward_graph, backward_nodes):
    mapper = _build_backward_mapper(device, initialise_data_extractor_zero=True)
    edge_attr, edge_index = backward_graph

    x_out = mapper(
        backward_nodes,
        batch_size=1,
        shard_info=BipartiteGraphShardInfo(),
        edge_attr=edge_attr,
        edge_index=edge_index,
    )

    assert torch.all(x_out == 0)
