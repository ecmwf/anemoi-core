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
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import pytest
import torch
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats
from torch_geometric.data import HeteroData
from omegaconf import OmegaConf

from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict
import nsight


from anemoi.models.utils.compile import mark_for_compilation


class CUDAGraphMapperWrapper(torch.nn.Module):
    """Wrapper that captures non-tensor args so make_graphed_callables sees only Tensors."""

    def __init__(self, mapper, batch_size, shard_shapes):
        super().__init__()
        self.mapper = mapper
        self.batch_size = batch_size
        self.shard_shapes = shard_shapes

    def forward(self, x_src, x_dst, edge_attr, edge_index):
        with torch.no_grad():
            return self.mapper.forward((x_src, x_dst), self.batch_size, self.shard_shapes, edge_attr, edge_index)


# Import the benchmark profiling function
import sys
from pathlib import Path
benchmark_utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(benchmark_utils_path))

LOGGER = logging.getLogger(__name__)


class GraphConfig_n320_to_o96:
    """Configuration for the graph used in benchmarks."""
    #num_src_nodes: int = 542080  
    #num_dst_nodes: int = 40320
    #num_edges: int = 748348  
    num_src_nodes: int = 542080
    num_dst_nodes: int = 40320 
    num_edges: int = 748348  

class GraphConfig_o96_to_n320:
    """Configuration for the graph used in benchmarks."""
    num_src_nodes: int = 40320
    num_dst_nodes: int = 542080
    num_edges: int = 1.62624e+06

def compile_config():
    return OmegaConf.create(
        {
            "compile": [
                {
                    "module": "anemoi.models.layers.conv.GraphTransformerConv",
                },
            ],
        }
    )

@dataclass
class MapperBenchmarkConfig:
    """Configuration for mapper benchmarks."""
    in_channels_src: int = 1024
    in_channels_dst: int = 1024
    trainable_size: int = 256
    hidden_dim: int = 256
    num_chunks: int = 4
    num_heads: int = 16
    mlp_hidden_ratio: int = 4
    qk_norm: bool = True
    cpu_offload: bool = False
    layer_kernels: field(default_factory=DotDict) = None
    shard_strategy: str = "heads"
    graph_attention_backend: str = "triton" # TODO pick triton by default if running on GPU
    edge_dim: int = None
    edge_pre_mlp: bool = False
    
    # Benchmark settings
    warmup_iter: int = 10
    run_iter: int = 50
    batch_size: int = 1

    def __post_init__(self):
        if self.layer_kernels is None:
            self.layer_kernels = load_layer_kernels(instance=False)
        
        

@pytest.fixture
def device():
    return "cuda:0"

@pytest.fixture
def mapper_benchmark_config():
    """Fixture providing benchmark configuration."""
    return MapperBenchmarkConfig()

def benchmark_graph(graph_config, device) -> HeteroData:
    """Create a realistic graph for benchmarking."""
    config = graph_config
    graph = HeteroData()
    
    # Create edge indices (ensure int types)
    num_src = int(config.num_src_nodes)
    num_dst = int(config.num_dst_nodes)
    num_edges = int(config.num_edges)
    
    graph[("nodes", "to", "nodes")].edge_index = torch.concat(
        [
            torch.randint(0, num_src, (1, num_edges), device=device),
            torch.randint(0, num_dst, (1, num_edges), device=device),
        ],
        axis=0,
    )
    
    # Add edge attributes (simulating real edge features)
    # TODO fix
    #   Source │ Target │  Num. edges │ Isolated Source │ Isolated Target │ Attribute dim │                     Attributes
   #───────┼────────┼─────────────┼─────────────────┼─────────────────┼───────────────┼───────────────────────────────
   #data   │ hidden │      748348 │            3064 │               0 │             3 │ edge_length(1D), edge_dirs(2D)
   #hidden │ data   │ 1.62624e+06 │               0 │               0 │             3 │ edge_length(1D), edge_dirs(2D)
   #───────┴────────┴─────────────┴─────────────────┴─────────────────┴───────────────┴───────────────────────────────
    graph[("nodes", "to", "nodes")].edge_attr1 = torch.rand((num_edges, 1), device=device)
    graph[("nodes", "to", "nodes")].edge_attr2 = torch.rand((num_edges, 32), device=device)
    
    return graph


def benchmark_graph_provider(graph_config, mapper_benchmark_config, device):
    """Create graph provider for benchmarking."""
    
    graph = benchmark_graph(graph_config, device)
    provider = create_graph_provider(
        graph=graph[("nodes", "to", "nodes")],
        edge_attributes=["edge_attr1", "edge_attr2"],
        src_size=graph_config.num_src_nodes,
        dst_size=graph_config.num_dst_nodes,
        trainable_size=mapper_benchmark_config.trainable_size,
    )
    return provider.to(device)

def dummy_loss(_out):
        loss = 0
        if isinstance(_out, torch.Tensor):
            loss += _out.sum()
        else:
            for o in _out:
                loss += o.sum()
        return loss

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("graph_config", [GraphConfig_n320_to_o96()])
@pytest.mark.parametrize("mode", ["both"])
@pytest.mark.parametrize("use_cuda_graphs", [False])
def test_benchmark_forward_mapper(mapper_benchmark_config, graph_config, benchmark_graph_provider, mode, device, use_cuda_graphs):
    """Benchmark the GraphTransformerForwardMapper."""
    config = mapper_benchmark_config
    
    LOGGER.debug("Benchmarking GraphTransformerForwardMapper")
    LOGGER.debug(f"Configuration: {asdict(config)}")
    
    if torch.cuda.is_available():
        # Reset memory stats
        reset_peak_memory_stats()
        empty_cache()
    gc.collect()

    torch.cuda.set_stream(torch.cuda.Stream())  # use a separate stream for graph capture to avoid synchronizing with other operations during warmup
    
    # Create mapper
    mapper_config = asdict(config)
    mapper_config["edge_dim"] = benchmark_graph_provider.edge_dim
    mapper_config["use_cuda_graphs"] = use_cuda_graphs
    mapper = GraphTransformerForwardMapper(**mapper_config).to(device)

    mapper = mark_for_compilation(mapper, compile_config().compile)
    
    # Create input tensors
    x_src = torch.rand(graph_config.num_src_nodes, config.in_channels_src, device=device, requires_grad=True)
    x_dst = torch.rand(graph_config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
    x = (x_src, x_dst)
    
    shard_shapes = ([list(x_src.shape)], [list(x_dst.shape)])
    edge_attr, edge_index, _ = benchmark_graph_provider.get_edges(batch_size=config.batch_size)



    #with nsight.annotate("Mapper forward (triton)"):
    #out = mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
    #torch.cuda.cudart().cudaProfilerStart()
    
    for use_cuda_graphs in [True, False]:
        print(f"Using cuda graphs: {use_cuda_graphs}")
        for i in range(10): # memory leak running more then 10 iterations for some reason, need to investigate
            x_src = torch.rand(graph_config.num_src_nodes, config.in_channels_src, device=device, requires_grad=True)
            x_dst = torch.rand(graph_config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
            x = (x_src, x_dst)
            torch.cuda.nvtx.range_push(f"iteration {i} start")
            start_time= time.time()

            if use_cuda_graphs:
                if i == 0: # warmup
                    with torch.cuda.stream(torch.cuda.Stream()): # run in a separate stream to avoid synchronizing the graph capture stream during warmup
                        out = mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
                        loss = dummy_loss(out)
                        with torch.autograd.grad_mode.set_multithreading_enabled(False): # dont run bwd pass in a seperarte thread
                            loss.backward()
                        del loss
                        end_time= time.time()
                        print(f"iteration {i}: {(end_time-start_time)*1000:.2f}ms")
                        torch.cuda.nvtx.range_pop()
                    continue
                if i == 1: #graph capture
                    x_src_static = torch.empty_like(x_src, device=device)
                    x_dst_static = torch.empty_like(x_dst, device=device)
                    x_static = (x_src_static, x_dst_static)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, capture_error_mode="thread_local"):  #dont error if a different thread does an unsupported cuda op
                        out = mapper.forward(x_static, config.batch_size, shard_shapes, edge_attr, edge_index)
                        loss = dummy_loss(out)
                        with torch.autograd.grad_mode.set_multithreading_enabled(False): # dont run bwd pass in a seperarte thread
                            loss.backward()

                #replay graph in subsequent iterations
                x_src_static.copy_(x_src, non_blocking=True)
                x_dst_static.copy_(x_dst, non_blocking=True)
                g.replay()
            else:
                out = mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
                loss = dummy_loss(out)
                with torch.autograd.grad_mode.set_multithreading_enabled(False): # dont run bwd pass in a seperarte thread
                    loss.backward()

            end_time= time.time()
            print(f"iteration {i}: {(end_time-start_time)*1000:.2f}ms")
            torch.cuda.nvtx.range_pop()


    #mapper_config["graph_attention_backend"] = 'pyg'
    #mapper_pyg = GraphTransformerForwardMapper(**mapper_config).to(device)
    #with nsight.annotate("Mapper forward (pyg)"):
    #    out = mapper_pyg.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("graph_config", [GraphConfig_o96_to_n320()])
@pytest.mark.parametrize("mode", ["both"])
def test_benchmark_backward_mapper(mapper_benchmark_config, graph_config, benchmark_graph_provider, mode, device):
    """Benchmark the GraphTransformerBackwardMapper."""
    config = mapper_benchmark_config
    
    LOGGER.debug("Benchmarking GraphTransformerBackwardMapper")
    LOGGER.debug(f"Configuration: {asdict(config)}")
    
    # Reset memory stats
    if torch.cuda.is_available():
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
    mapper = mark_for_compilation(mapper, compile_config().compile)
    
    # Create input tensors (note: backward mapper has different input channel requirements)
    x_src = torch.rand(graph_config.num_src_nodes, config.hidden_dim, device=device, requires_grad=True)
    x_dst = torch.rand(graph_config.num_dst_nodes, config.in_channels_dst, device=device, requires_grad=True)
    x = (x_src, x_dst)
    
    shard_shapes = ([list(x_src.shape)], [list(x_dst.shape)])
    edge_attr, edge_index, _ = benchmark_graph_provider.get_edges(batch_size=config.batch_size)
    
    # Define forward function for benchmarking
    def forward_fn():
        return mapper.forward(x, config.batch_size, shard_shapes, edge_attr, edge_index)
    
    # Run benchmark
    LOGGER.debug(f"Running {mode} benchmark...")
    #run_time, peak_memory = run_benchmark(
    #    forward_fn,
    #    mode=mode,
    #    warmup_iter=config.warmup_iter,
    #    run_iter=config.run_iter
    #)
    
    LOGGER.debug(f"Backward mapper benchmark complete:")
    LOGGER.debug(f"  Time per iteration: {run_time:.2f} ms")
    LOGGER.debug(f"  Peak memory usage: {peak_memory:.2f} MB")
    
    assert run_time > 0, "Benchmark did not run successfully"
    assert peak_memory > 0, "Benchmark did not measure memory usage"
    
#@pytest.mark.gpu
#@pytest.mark.slow
#@pytest.mark.parametrize("graph_config", [GraphConfig_o96_to_n320()])
#@pytest.mark.parametrize("mode", ['both'])
#@pytest.mark.parametrize("backend", ['pyg', 'triton'])
#def test_benchmark_forward_mapper_different_backends(mapper_benchmark_config, graph_config, benchmark_graph_provider, mode, backend, device):
#    """Benchmark the GraphTransformerForwardMapper with different attention backends."""
#    LOGGER.debug(f"Benchmarking with {backend} backend...")
#    if (not device.startswith("cuda")) and backend == "triton":
#        pytest.skip("Triton backend is not supported on CPU.")
#    mapper_benchmark_config.graph_attention_backend = backend
#    use_cuda_graphs = backend == False
#    test_benchmark_forward_mapper(mapper_benchmark_config, graph_config, benchmark_graph_provider, mode, device, use_cuda_graphs)
    
#@pytest.mark.gpu
#@pytest.mark.slow
#@pytest.mark.parametrize("graph_config", [GraphConfig_o96_to_n320()])
#@pytest.mark.parametrize("mode", ['both'])
#@pytest.mark.parametrize("num_chunks", [1,2,4,8])
#def test_benchmark_forward_mapper_different_num_chunks(mapper_benchmark_config, graph_config, benchmark_graph_provider, mode, num_chunks, device):
#    """Benchmark the GraphTransformerForwardMapper with different num_chunks."""
#    LOGGER.debug(f"Benchmarking with num_chunks={num_chunks}...")
#    mapper_benchmark_config.num_chunks = num_chunks
#    use_cuda_graphs = False
#    test_benchmark_forward_mapper(mapper_benchmark_config, graph_config, benchmark_graph_provider, mode, device, use_cuda_graphs)

#@nsight.analyze.plot(
    #"mapper_fwd_n320_o96.png",
   # plot_type="bar",
    #title="Mapper Forward Pass Performance (n320 to o96)",
#    ylabel="Time (ms)"#,
#)
#@nsight.analyze.kernel(
#    configs=[512, 1024, 2048],
#    runs=3,
    #metrics=["dram__throughput.avg"],
    #metrics=["sm__cycles_elapsed.avg",],
    #metrics=["gpu__time_duration.sum",],
    #metrics=["smsp__sass_inst_executed_op_shared.sum",],
#    metrics=["dram__cycles_active_read.sum",],
#    #replay_mode='range', # annotation should only have 1 kernel # didnt work for some reason
#    combine_kernel_metrics=lambda x, y: x + y,  # Sum metrics from multiple kernels
#)
def benchmark_forward_mapper(num_channels):
    mapper = MapperBenchmarkConfig(in_channels_src=num_channels, in_channels_dst=num_channels)
    graph=benchmark_graph_provider(GraphConfig_n320_to_o96(), mapper, "cuda:0")
    return test_benchmark_forward_mapper(mapper,GraphConfig_n320_to_o96(),graph, "both", "cuda:0", False)

def main() -> None:
    graph=benchmark_graph_provider(GraphConfig_n320_to_o96(), MapperBenchmarkConfig(), "cuda:0")
    result = benchmark_forward_mapper(1024)
    #print(result.to_dataframe())
    #print("\nTip: Run 'ncu --query-metrics' to see all available metrics!")


if __name__ == "__main__":
    main()
