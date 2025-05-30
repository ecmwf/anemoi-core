# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib

import pytest
import torch
import torch.nn as nn

import anemoi.models.layers.block
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.block import GraphTransformerProcessorBlock
from anemoi.models.layers.conv import GraphTransformerConv
from anemoi.models.layers.utils import load_layer_kernels


@pytest.fixture
def init():
    in_channels = 128
    hidden_dim = 64
    out_channels = 128
    edge_dim = 11
    bias = True
    num_heads = 8
    num_chunks = 2
    layer_kernels = load_layer_kernels()
    qk_norm = True
    return (
        in_channels,
        hidden_dim,
        out_channels,
        edge_dim,
        layer_kernels,
        bias,
        num_heads,
        num_chunks,
        qk_norm,
    )


@pytest.fixture
def block(init):
    (
        in_channels,
        hidden_dim,
        out_channels,
        edge_dim,
        layer_kernels,
        bias,
        num_heads,
        num_chunks,
        qk_norm,
    ) = init
    return GraphTransformerProcessorBlock(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        edge_dim=edge_dim,
        layer_kernels=layer_kernels,
        num_heads=num_heads,
        bias=bias,
        update_src_nodes=False,
        num_chunks=num_chunks,
        qk_norm=qk_norm,
    )


def test_GraphTransformerProcessorBlock_init(init, block):
    (
        _in_channels,
        _hidden_dim,
        out_channels,
        _edge_dim,
        _layer_kernels,
        _bias,
        num_heads,
        num_chunks,
        _qk_norm,
    ) = init
    assert isinstance(
        block, GraphTransformerProcessorBlock
    ), "block is not an instance of GraphTransformerProcessorBlock"
    assert (
        block.out_channels_conv == out_channels // num_heads
    ), f"block.out_channels_conv ({block.out_channels_conv}) != out_channels // num_heads ({out_channels // num_heads})"
    assert block.num_heads == num_heads, f"block.num_heads ({block.num_heads}) != num_heads ({num_heads})"
    assert block.num_chunks == num_chunks, f"block.num_chunks ({block.num_chunks}) != num_chunks ({num_chunks})"
    assert isinstance(block.lin_key, torch.nn.Linear), "block.lin_key is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_query, torch.nn.Linear), "block.lin_query is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_value, torch.nn.Linear), "block.lin_value is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_self, torch.nn.Linear), "block.lin_self is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_edge, torch.nn.Linear), "block.lin_edge is not an instance of torch.nn.Linear"
    assert isinstance(block.conv, GraphTransformerConv), "block.conv is not an instance of GraphTransformerConv"
    assert isinstance(block.projection, torch.nn.Linear), "block.projection is not an instance of torch.nn.Linear"
    assert isinstance(
        block.node_dst_mlp, torch.nn.Sequential
    ), "block.node_dst_mlp is not an instance of torch.nn.Sequential"
    assert block.q_norm.bias is None
    assert block.k_norm.bias is None


def test_GraphTransformerProcessorBlock_shard_qkve_heads(init, block):
    (
        in_channels,
        _hidden_dim,
        _out_channels,
        _edge_dim,
        _layer_kernels,
        _bias,
        num_heads,
        _num_chunks,
        _qk_norm,
    ) = init
    query = torch.randn(in_channels, num_heads * block.out_channels_conv)
    key = torch.randn(in_channels, num_heads * block.out_channels_conv)
    value = torch.randn(in_channels, num_heads * block.out_channels_conv)
    edges = torch.randn(in_channels, num_heads * block.out_channels_conv)
    shapes = (10, 10, 10)
    batch_size = 1
    query, key, value, edges = block.shard_qkve_heads(query, key, value, edges, shapes, batch_size)
    assert query.shape == (in_channels, num_heads, block.out_channels_conv)
    assert key.shape == (in_channels, num_heads, block.out_channels_conv)
    assert value.shape == (in_channels, num_heads, block.out_channels_conv)
    assert edges.shape == (in_channels, num_heads, block.out_channels_conv)


def test_GraphTransformerProcessorBlock_shard_output_seq(init, block):
    (
        in_channels,
        _hidden_dim,
        _out_channels,
        _edge_dim,
        _layer_kernels,
        _bias,
        num_heads,
        _num_chunks,
        _qk_norm,
    ) = init
    out = torch.randn(in_channels, num_heads, block.out_channels_conv)
    shapes = (10, 10, 10)
    batch_size = 1
    out = block.shard_output_seq(out, shapes, batch_size)
    assert out.shape == (in_channels, num_heads * block.out_channels_conv)


@pytest.mark.gpu
def test_GraphTransformerProcessorBlock_forward_backward(init, block):
    (
        in_channels,
        _hidden_dim,
        out_channels,
        edge_dim,
        _layer_kernels,
        _bias,
        _num_heads,
        _num_chunks,
        _qk_norm,
    ) = init

    # Generate random input tensor
    x = torch.randn((10, in_channels))
    edge_attr = torch.randn((10, edge_dim))
    edge_index = torch.randint(1, 10, (2, 10))
    shapes = (10, 10, 10)
    batch_size = 1
    size = 10

    # Forward pass
    output, _ = block(x, edge_attr, edge_index, shapes, batch_size, size)

    # Check output shape
    assert output.shape == (10, out_channels)

    # Generate dummy target and loss function
    target = torch.randn((10, out_channels))
    loss_fn = nn.MSELoss()

    # Compute loss
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Check gradients
    for param in block.parameters():
        assert param.grad is not None, f"param.grad is None for {param}"
        assert (
            param.grad.shape == param.shape
        ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"


@pytest.fixture
def test_GraphTransformerProcessorBlock_chunking(init, block, monkeypatch):
    (
        in_channels,
        _hidden_dim,
        _out_channels,
        edge_dim,
        _bias,
        _activation,
        _num_heads,
        _num_chunks,
    ) = init
    # Initialize GraphTransformerProcessorBlock
    block = block

    # Generate random input tensor
    x = torch.randn((10, in_channels))
    edge_attr = torch.randn((10, edge_dim))
    edge_index = torch.randint(1, 10, (2, 10))
    shapes = (10, 10, 10)
    batch_size = 1
    size = 10
    num_chunks = torch.randint(2, 10, (1,)).item()

    # manually set to non-training mode
    block.eval()

    # result with chunks
    monkeypatch.setenv("ANEMOI_INFERENCE_NUM_CHUNKS", str(num_chunks))
    importlib.reload(anemoi.models.layers.block)
    out_chunked, _ = block(x, edge_attr, edge_index, shapes, batch_size, size)
    # result without chunks, reload block for new env variable
    monkeypatch.setenv("ANEMOI_INFERENCE_NUM_CHUNKS", "1")
    importlib.reload(anemoi.models.layers.block)
    out, _ = block(x, edge_attr, edge_index, shapes, batch_size, size)

    assert out.shape == out_chunked.shape, f"out.shape ({out.shape}) != out_chunked.shape ({out_chunked.shape})"
    assert torch.allclose(out, out_chunked, atol=1e-4), "out != out_chunked"


@pytest.fixture
def mapper_block(init):
    (
        in_channels,
        hidden_dim,
        out_channels,
        edge_dim,
        layer_kernels,
        bias,
        num_heads,
        num_chunks,
        qk_norm,
    ) = init
    return GraphTransformerMapperBlock(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        edge_dim=edge_dim,
        layer_kernels=layer_kernels,
        num_heads=num_heads,
        bias=bias,
        update_src_nodes=False,
        num_chunks=num_chunks,
        qk_norm=qk_norm,
    )


def test_GraphTransformerMapperBlock_init(init, mapper_block):
    (
        _in_channels,
        _hidden_dim,
        out_channels,
        _edge_dim,
        _layer_kernels,
        _bias,
        num_heads,
        num_chunks,
        _qk_norm,
    ) = init
    block = mapper_block
    assert isinstance(block, GraphTransformerMapperBlock), "block is not an instance of GraphTransformerMapperBlock"
    assert (
        block.out_channels_conv == out_channels // num_heads
    ), f"block.out_channels_conv ({block.out_channels_conv}) != out_channels // num_heads ({out_channels // num_heads})"
    assert block.num_heads == num_heads, f"block.num_heads ({block.num_heads}) != num_heads ({num_heads})"
    assert block.num_chunks == num_chunks, f"block.num_chunks ({block.num_chunks}) != num_chunks ({num_chunks})"
    assert isinstance(block.lin_key, torch.nn.Linear), "block.lin_key is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_query, torch.nn.Linear), "block.lin_query is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_value, torch.nn.Linear), "block.lin_value is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_self, torch.nn.Linear), "block.lin_self is not an instance of torch.nn.Linear"
    assert isinstance(block.lin_edge, torch.nn.Linear), "block.lin_edge is not an instance of torch.nn.Linear"
    assert isinstance(block.conv, GraphTransformerConv), "block.conv is not an instance of GraphTransformerConv"
    assert isinstance(block.projection, torch.nn.Linear), "block.projection is not an instance of torch.nn.Linear"
    assert isinstance(
        block.node_dst_mlp, torch.nn.Sequential
    ), "block.node_dst_mlp is not an instance of torch.nn.Sequential"


def test_GraphTransformerMapperBlock_shard_qkve_heads(init, mapper_block):
    (
        in_channels,
        _hidden_dim,
        _out_channels,
        _edge_dim,
        _layer_kernels,
        _bias,
        num_heads,
        _num_chunks,
        _qk_norm,
    ) = init
    block = mapper_block
    query = torch.randn(in_channels, num_heads * block.out_channels_conv)
    key = torch.randn(in_channels, num_heads * block.out_channels_conv)
    value = torch.randn(in_channels, num_heads * block.out_channels_conv)
    edges = torch.randn(in_channels, num_heads * block.out_channels_conv)
    shapes = (10, 10, 10)
    batch_size = 1
    query, key, value, edges = block.shard_qkve_heads(query, key, value, edges, shapes, batch_size)
    assert query.shape == (in_channels, num_heads, block.out_channels_conv)
    assert key.shape == (in_channels, num_heads, block.out_channels_conv)
    assert value.shape == (in_channels, num_heads, block.out_channels_conv)
    assert edges.shape == (in_channels, num_heads, block.out_channels_conv)


def test_GraphTransformerMapperBlock_shard_output_seq(init, mapper_block):
    (
        in_channels,
        _hidden_dim,
        _out_channels,
        _edge_dim,
        _layer_kernels,
        _bias,
        num_heads,
        _num_chunks,
        _qk_norm,
    ) = init
    block = mapper_block
    out = torch.randn(in_channels, num_heads, block.out_channels_conv)
    shapes = (10, 10, 10)
    batch_size = 1
    out = block.shard_output_seq(out, shapes, batch_size)
    assert out.shape == (in_channels, num_heads * block.out_channels_conv)


def test_GraphTransformerMapperBlock_forward_backward(init, mapper_block):
    (
        in_channels,
        _hidden_dim,
        out_channels,
        edge_dim,
        _layer_kernels,
        _bias,
        _num_heads,
        _num_chunks,
        _qk_norm,
    ) = init
    # Initialize GraphTransformerMapperBlock
    block = mapper_block

    # Generate random input tensor
    x = (torch.randn((10, in_channels)), torch.randn((10, in_channels)))
    edge_attr = torch.randn((10, edge_dim))
    edge_index = torch.randint(1, 10, (2, 10))
    shapes = (10, 10, 10)
    batch_size = 1
    size = (10, 10)

    # Forward pass
    output, _ = block(x, edge_attr, edge_index, shapes, batch_size, size)

    # Check output shape
    assert output[0].shape == (10, out_channels)
    assert output[1].shape == (10, out_channels)

    # Generate dummy target and loss function
    target = torch.randn((10, out_channels))
    loss_fn = nn.MSELoss()

    # Compute loss
    loss_dst = loss_fn(output[1], target)

    # Backward pass
    loss_dst.backward()

    # Check gradients
    for param in block.parameters():
        assert param.grad is not None, f"param.grad is None for {param}"
        assert (
            param.grad.shape == param.shape
        ), f"param.grad.shape ({param.grad.shape}) != param.shape ({param.shape}) for {param}"


def test_GraphTransformerMapperBlock_chunking(init, mapper_block, monkeypatch):
    (
        in_channels,
        _hidden_dim,
        _out_channels,
        edge_dim,
        _layer_kernels,
        _bias,
        _num_heads,
        _num_chunks,
        _qk_norm,
    ) = init
    # Initialize GraphTransformerMapperBlock
    block = mapper_block

    # Generate random input tensor
    x = (torch.randn((10, in_channels)), torch.randn((10, in_channels)))
    edge_attr = torch.randn((10, edge_dim))
    edge_index = torch.randint(1, 10, (2, 10))
    shapes = (10, 10, 10)
    batch_size = 1
    size = (10, 10)
    num_chunks = torch.randint(2, 10, (1,)).item()

    # manually set to non-training mode
    block.eval()

    # result with chunks
    monkeypatch.setenv("ANEMOI_INFERENCE_NUM_CHUNKS", str(num_chunks))
    importlib.reload(anemoi.models.layers.block)
    out_chunked, _ = block(x, edge_attr, edge_index, shapes, batch_size, size=size)
    # result without chunks, reload block for new env variable
    monkeypatch.setenv("ANEMOI_INFERENCE_NUM_CHUNKS", "1")
    importlib.reload(anemoi.models.layers.block)
    out, _ = block(x, edge_attr, edge_index, shapes, batch_size, size=size)

    assert out[0].shape == out_chunked[0].shape, f"out.shape ({out.shape}) != out_chunked.shape ({out_chunked.shape})"
    assert out[1].shape == out_chunked[1].shape, f"out.shape ({out.shape}) != out_chunked.shape ({out_chunked.shape})"
    assert torch.allclose(out[0], out_chunked[0], atol=1e-4), "out != out_chunked"
    assert torch.allclose(out[1], out_chunked[1], atol=1e-4), "out != out_chunked"
