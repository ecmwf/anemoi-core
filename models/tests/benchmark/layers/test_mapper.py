# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import gc
import logging
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch
from torch import nn
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

# Import the benchmark profiling function
import sys
from pathlib import Path
benchmark_utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(benchmark_utils_path))
from profiling import benchmark as run_benchmark

LOGGER = logging.getLogger(__name__)


@dataclass
class MapperBenchmarkConfig:
    """Configuration for mapper benchmarks."""
    in_channels_src: int = 1024
    in_channels_dst: int = 1024
    hidden_dim: int = 256
    num_chunks: int = 4
    num_heads: int = 16
    mlp_hidden_ratio: int = 4
    qk_norm: bool = True
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None
    shard_strategy: str = "edges"
    graph_attention_backend: str = "triton"
    edge_dim: int = None
    edge_pre_mlp: bool = False
    
    # Graph configuration
    num_src_nodes: int = 40320  # Realistic number of grid points
    num_dst_nodes: int = 40320
    num_edges: int = 500000  # Realistic number of edges
    
    # Benchmark settings
    warmup_iter: int = 10
    run_iter: int = 50
    batch_size: int = 1

    def __post_init__(self):
        if self.layer_kernels is None:
            self.layer_kernels = load_layer_kernels(instance=False)


@pytest.fixture
def mapper_benchmark_config():
    """Fixture providing benchmark configuration."""
    return MapperBenchmarkConfig()


@pytest.fixture
def benchmark_graph(mapper_benchmark_config, device="cuda:0") -> HeteroData:
    """Create a realistic graph for benchmarking."""
    config = mapper_benchmark_config
    graph = HeteroData()
    
    # Create edge indices
    graph[("nodes", "to", "nodes")].edge_index = torch.concat(
        [
            torch.randint(0, config.num_src_nodes, (1, config.num_edges), device=device),
            torch.randint(0, config.num_dst_nodes, (1, config.num_edges), device=device),
        ],
        axis=0,
    )
    
    # Add edge attributes (simulating real edge features)
    graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((config.num_edges, 1), device=device)
    graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((config.num_edges, 32), device=device)
    
    return graph


@pytest.fixture
def benchmark_graph_provider(benchmark_graph, mapper_benchmark_config, device="cuda:0"):
    """Create graph provider for benchmarking."""
    config = mapper_benchmark_config
    provider = create_graph_provider(
        graph=benchmark_graph[("nodes", "to", "nodes")],
        edge_attributes=["edge_attr1", "edge_attr2"],
        src_size=config.num_src_nodes,
        dst_size=config.num_dst_nodes,
        trainable_size=6,
    )
    return provider.to(device)


@pytest.mark.gpu
@pytest.mark.slow
def test_benchmark_forward_mapper(mapper_benchmark_config, benchmark_graph_provider):
    """Benchmark the GraphTransformerForwardMapper."""
    config = mapper_benchmark_config
    device = "cuda:0"
    
    LOGGER.info("Benchmarking GraphTransformerForwardMapper")
    LOGGER.info(f"Configuration: {asdict(config)}")
    
    # Reset memory stats
    reset_peak_memory_stats()
    empty_cache()
    gc.collect()
    
    # Create mapper
    mapper_config = asdict(config)
    mapper_config["edge_dim"] = benchmark_graph_provider.edge_dim
    mapper = GraphTransformerForwardMapper(**mapper_config).to(device)
    
    # Create input tensors
    x_src = torch.rand(config.num_src_nodes, config.in_channels_src, device=device, requires_grad=True)
    x_dst = torch.rand(config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
    x = (x_src, x_dst)
    
    shard_shapes = ([list(x_src.shape)], [list(x_dst.shape)])
    edge_attr, edge_index, _ = benchmark_graph_provider.get_edges(batch_size=config.batch_size)
    
    # Define forward function for benchmarking
    def forward_fn():
        return mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
    
    # Run benchmark
    LOGGER.info("Running forward+backward benchmark...")
    run_time, peak_memory = run_benchmark(
        forward_fn,
        mode="both",
        warmup_iter=config.warmup_iter,
        run_iter=config.run_iter
    )
    
    LOGGER.info(f"Forward mapper benchmark complete:")
    LOGGER.info(f"  Time per iteration: {run_time:.2f} ms")
    LOGGER.info(f"  Peak memory usage: {peak_memory:.2f} MB")
    
    # Assertions to ensure benchmark ran successfully
    assert run_time > 0, "Benchmark time should be positive"
    assert peak_memory > 0, "Peak memory should be positive"


@pytest.mark.gpu
@pytest.mark.slow
def test_benchmark_backward_mapper(mapper_benchmark_config, benchmark_graph_provider):
    """Benchmark the GraphTransformerBackwardMapper."""
    config = mapper_benchmark_config
    device = "cuda:0"
    
    LOGGER.info("Benchmarking GraphTransformerBackwardMapper")
    LOGGER.info(f"Configuration: {asdict(config)}")
    
    # Reset memory stats
    reset_peak_memory_stats()
    empty_cache()
    gc.collect()
    
    # Create mapper
    mapper_config = asdict(config)
    mapper_config["edge_dim"] = benchmark_graph_provider.edge_dim
    out_channels_dst = 128  # Output channels for backward mapper
    mapper = GraphTransformerBackwardMapper(
        **mapper_config,
        out_channels_dst=out_channels_dst
    ).to(device)
    
    # Create input tensors (note: backward mapper has different input channel requirements)
    x_src = torch.rand(config.num_src_nodes, config.hidden_dim, device=device, requires_grad=True)
    x_dst = torch.rand(config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
    x = (x_src, x_dst)
    
    shard_shapes = ([list(x_src.shape)], [list(x_dst.shape)])
    edge_attr, edge_index, _ = benchmark_graph_provider.get_edges(batch_size=config.batch_size)
    
    # Define forward function for benchmarking
    def forward_fn():
        return mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
    
    # Run benchmark
    LOGGER.info("Running forward+backward benchmark...")
    run_time, peak_memory = run_benchmark(
        forward_fn,
        mode="both",
        warmup_iter=config.warmup_iter,
        run_iter=config.run_iter
    )
    
    LOGGER.info(f"Backward mapper benchmark complete:")
    LOGGER.info(f"  Time per iteration: {run_time:.2f} ms")
    LOGGER.info(f"  Peak memory usage: {peak_memory:.2f} MB")
    
    # Assertions to ensure benchmark ran successfully
    assert run_time > 0, "Benchmark time should be positive"
    assert peak_memory > 0, "Peak memory should be positive"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("backend", ["triton", "pyg"])
def test_benchmark_mapper_backends(mapper_benchmark_config, benchmark_graph_provider, backend):
    """Compare performance across different backends."""
    config = mapper_benchmark_config
    config.graph_attention_backend = backend
    device = "cuda:0"
    
    LOGGER.info(f"Benchmarking ForwardMapper with backend: {backend}")
    
    # Reset memory stats
    reset_peak_memory_stats()
    empty_cache()
    gc.collect()
    
    # Create mapper
    mapper_config = asdict(config)
    mapper_config["edge_dim"] = benchmark_graph_provider.edge_dim
    mapper = GraphTransformerForwardMapper(**mapper_config).to(device)
    
    # Create input tensors
    x_src = torch.rand(config.num_src_nodes, config.in_channels_src, device=device, requires_grad=True)
    x_dst = torch.rand(config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
    x = (x_src, x_dst)
    
    shard_shapes = ([list(x_src.shape)], [list(x_dst.shape)])
    edge_attr, edge_index, _ = benchmark_graph_provider.get_edges(batch_size=config.batch_size)
    
    # Define forward function for benchmarking
    def forward_fn():
        return mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
    
    # Run benchmark
    run_time, peak_memory = run_benchmark(
        forward_fn,
        mode="both",
        warmup_iter=config.warmup_iter,
        run_iter=config.run_iter
    )
    
    LOGGER.info(f"{backend} backend benchmark complete:")
    LOGGER.info(f"  Time per iteration: {run_time:.2f} ms")
    LOGGER.info(f"  Peak memory usage: {peak_memory:.2f} MB")
    
    # Assertions
    assert run_time > 0, "Benchmark time should be positive"
    assert peak_memory > 0, "Peak memory should be positive"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("num_chunks", [1, 2, 4, 8])
def test_benchmark_mapper_chunking(mapper_benchmark_config, benchmark_graph_provider, num_chunks):
    """Benchmark mapper with different chunking strategies."""
    config = mapper_benchmark_config
    config.num_chunks = num_chunks
    device = "cuda:0"
    
    LOGGER.info(f"Benchmarking ForwardMapper with num_chunks: {num_chunks}")
    
    # Reset memory stats
    reset_peak_memory_stats()
    empty_cache()
    gc.collect()
    
    # Create mapper
    mapper_config = asdict(config)
    mapper_config["edge_dim"] = benchmark_graph_provider.edge_dim
    mapper = GraphTransformerForwardMapper(**mapper_config).to(device)
    
    # Create input tensors
    x_src = torch.rand(config.num_src_nodes, config.in_channels_src, device=device, requires_grad=True)
    x_dst = torch.rand(config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
    x = (x_src, x_dst)
    
    shard_shapes = ([list(x_src.shape)], [list(x_dst.shape)])
    edge_attr, edge_index, _ = benchmark_graph_provider.get_edges(batch_size=config.batch_size)
    
    # Define forward function for benchmarking
    def forward_fn():
        return mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
    
    # Run benchmark
    run_time, peak_memory = run_benchmark(
        forward_fn,
        mode="both",
        warmup_iter=config.warmup_iter,
        run_iter=config.run_iter
    )
    
    LOGGER.info(f"Chunking={num_chunks} benchmark complete:")
    LOGGER.info(f"  Time per iteration: {run_time:.2f} ms")
    LOGGER.info(f"  Peak memory usage: {peak_memory:.2f} MB")
    
    # Assertions
    assert run_time > 0, "Benchmark time should be positive"
    assert peak_memory > 0, "Peak memory should be positive"


