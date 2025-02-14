from anemoi.models.layers.attention import BlockMaskManager, flex_attention, create_block_mask
from torch.nn.attention.flex_attention import flex_attention
import torch
from torch.nn.attention.flex_attention import create_block_mask
import os
import triton
import triton.tools.experimental_descriptor
import triton.language as tl




try:
    from flash_attn.flash_attn_interface import \
        flash_attn_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False
    
print(f"{HAS_FLASH=}")

#use_flash=True
##if use_flash:
#    out = flash_attn_func(query, key, value, window_size=self.window_size)

flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    fullgraph=os.getenv("FULL_GRAPH", "1") == "1",
    #    mode="max-autotune",
    #    options={"shape_padding": True}
)
print("compiled flex_attn") #why do we compile bash flex attn

# create_block_mask = torch.compile(create_block_mask, dynamic=True, fullgraph=True) #https://twitter.com/cHHillee/status/1851418255749169419
create_block_mask = torch.compile(
    create_block_mask,
    dynamic=False,
    fullgraph=os.getenv("FULL_GRAPH", "1") == "1",
    #    mode="max-autotune",
    #    options={"shape_padding": True}
)  # https://twitter.com/cHHillee/status/1851418255749169419ate
#print("compiled flex_attn")
    
#processor_block_mask:
#  method: knn_haversine # knn_adjacency_matrix, knn_index_matrix, haversine, knn_haversine
#  base_grid: query
#  attention_span: 160
#  block_size: [128,128]
from torch_geometric.data import HeteroData
from pathlib import Path
import logging
from anemoi.utils.config import DotDict
LOGGER = logging.getLogger(__name__)

print("finished importing")

def graph_data(config) -> HeteroData:
    """Graph data.

    Creates the graph in all workers.
    """

    from anemoi.graphs.create import GraphCreator

    print("creating graph")
    result = GraphCreator(config=config).create(
        save_path="/ec/res4/scratch/naco/tmp/knn-flex/flex.graph",
        overwrite=False,
    )
    print("created graph")
    return result
    
def create_block_mask(gconfig):
    _graph_name_hidden=gconfig.hidden
    _graph_name_data=gconfig.data
    processor_grid_name = _graph_name_hidden
    _graph_data=graph_data(gconfig)
    
    #based on training/src/anemoi/training/config/model/transformer_flex.yaml  processor block mask

    config_model_processor_block_mask= DotDict({'method': 'knn_haversine', 'base_grid': 'query', 'attention_span': 608, 'block_size': [128, 128]})
    print("creating block mask")
    bmc = BlockMaskManager(
        _graph_data,
        **config_model_processor_block_mask,
        query_grid_name=processor_grid_name,
        keyvalue_grid_name=processor_grid_name,
        base_attention_span_grid=_graph_name_data,)
    print("created block mask")
    return bmc

#query=torch.rand([1, 16, 10242, 64], device="cuda", dtype=torch.float32)
#key=torch.rand([1, 16, 10242, 64], device="cuda", dtype=torch.float32)
#value=torch.rand([1, 16, 10242, 64], device="cuda", dtype=torch.float32)
print("created input tensors")

kernel_options=DotDict({'BLOCK_M': 64, 'BLOCK_N': 64, 'HAS_FULL_BLOCKS': 1, 'ROWS_GUARANTEED_SAFE': True})

#HAS_FLASH=False
TORCH_HAS_FP8 = False # hasattr(torch, 'float8_e5m2')
NUM_CHANNELS=1024
BATCH, N_HEADS = 1, 16
HEAD_DIM= NUM_CHANNELS // N_HEADS #64 by default
DEVICE="cuda"
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
#for mode in ["fwd"]:
    for causal in [False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(8, 12)], #256 -> 4096
                line_arg="provider",
                line_vals=["flex-fp16", "flex-k"] + (["flex-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash-sw"] if HAS_FLASH else []),
                line_names=["FlexAttn", "FlexAttn (w kernel Options)"] + (["Flex [FP8]"] if TORCH_HAS_FP8 else []) +
                (["FlashAttn"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                #ylabel="TFLOPS",
                ylabel="Time (ms)",
                xlabel="#Channels",
                plot_name=f"flex-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    #"HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))
import functools

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, causal, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    #[1, 16, 1024, 64]
    HEAD_DIM=N_CTX//H
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    if "flex" in provider:
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        kernel_options=DotDict()
        if "k" in provider:
            kernel_options = DotDict({'BLOCK_M': 64, 'BLOCK_N': 64, 'HAS_FULL_BLOCKS': 1, 'ROWS_GUARANTEED_SAFE': True})
        #fn = lambda: attention(q, k, v, causal, sm_scale)
        #flex_attention=torch.compile()
        #flex_attention = torch.compile(functools.partial(flex_attention, block_mask=block_mask.get_block_mask(device=q.device))) 
        
        fn = lambda: flex_attention(q, k, v, block_mask=block_mask.get_block_mask(device=q.device), kernel_options=kernel_options)
        #fn = lambda: flex_attention(q, k, v, kernel_options=kernel_options)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash-sw":
        fn = lambda: flash_attn_func(q, k, v, window_size=(512,512), causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    #return total_flops * 1e-12 / (ms * 1e-3)
    return ms


if __name__ == "__main__":
    graph_config=DotDict({'overwrite': True, 'data': 'data', 'hidden': 'hidden', 'nodes': DotDict({'data': DotDict({'node_builder': DotDict({'_target_': 'anemoi.graphs.nodes.ZarrDatasetNodes', 'dataset': '/home/mlx/ai-ml/datasets//aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v7.zarr'}), 'attributes': DotDict({'area_weight': DotDict({'_target_': 'anemoi.graphs.nodes.attributes.AreaWeights', 'norm': 'unit-max'})})}), 'hidden': DotDict({'node_builder': DotDict({'_target_': 'anemoi.graphs.nodes.TriNodes', '_convert_': 'partial', 'resolution': 5}), 'attributes': DotDict({'area_weight': DotDict({'_target_': 'anemoi.graphs.nodes.attributes.AreaWeights', 'norm': 'unit-max'})})})}), 'edges': [], 'attributes': DotDict({'nodes': DotDict({'area_weight': DotDict({'_target_': 'anemoi.graphs.nodes.attributes.AreaWeights', 'norm': 'unit-max'})}), 'edges': DotDict({'edge_length': DotDict({'_target_': 'anemoi.graphs.edges.attributes.EdgeLength', 'norm': 'unit-std'}), 'edge_dirs': DotDict({'_target_': 'anemoi.graphs.edges.attributes.EdgeDirection', 'norm': 'unit-std'})})})})

    block_mask=create_block_mask(graph_config)
    kernel_options={}

    bench_flash_attention.run(save_path="./bm-results", print_data=True, show_plots=True,)
