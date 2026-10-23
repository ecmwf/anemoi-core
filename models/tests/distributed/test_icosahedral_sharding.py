# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A processor with icosahedral neighbourhood attention gives the same results with the points split across GPUs.

Every rank builds the same processor and inputs, runs it once on its own over the whole mesh (the
reference) and once with the points split across the ranks, each rank fetching its halo before
every layer, and compares outputs, input gradients and the parameter gradients summed over the
ranks. On CPU (gloo) in float64 with the dense-mask backend; on GPU (nccl) in float32 with the
Triton kernels.

Run with ``pytest --distributed [--distributed-world-size N] [--distributed-backend nccl]``.
"""

import pytest
import torch
import torch.distributed as dist

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.processor import TransformerProcessor
from anemoi.models.layers.utils import load_layer_kernels
from tests.distributed._distributed_runner import _run_distributed_test
from tests.layers.test_icosahedral_grid import icosahedron_coords

NUM_CHANNELS = 32
COND_CHANNELS = 3
RESOLUTION = 3

VARIANTS = {
    "radius 2, balanced": dict(radius=2, conditional=False, split="balanced"),
    "radius 3, balanced, conditional": dict(radius=3, conditional=True, split="balanced"),
    "radius 2, uneven, conditional": dict(radius=2, conditional=True, split="uneven"),
}


def split_points(num_points: int, world_size: int, how: str) -> list[int]:
    """Number of points per rank: ``balanced`` as the model splits, or ``uneven`` with one tiny shard."""
    if how == "balanced":
        return get_balanced_partition_sizes(num_points, world_size)
    cuts = [3] + [round(num_points * (k / world_size) ** 1.5) for k in range(2, world_size)]
    edges = [0, *cuts, num_points]
    return [b - a for a, b in zip(edges[:-1], edges[1:])]


def _build(radius: int, backend: str, conditional: bool) -> TransformerProcessor:
    kernels = {}
    if conditional:
        kernels = {
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "condition_shape": COND_CHANNELS,
                "zero_init": False,
            }
        }
    layer = TransformerProcessor(
        num_layers=2,
        num_chunks=2,
        num_channels=NUM_CHANNELS,
        num_heads=2,  # head dim 16, the smallest the Triton kernels take
        mlp_hidden_ratio=2,
        qk_norm=True,
        attention_implementation="neighbourhood",
        neighbourhood={"grid": "icosahedral", "radius": radius, "backend": backend},
        rotary_embeddings={"max_frequency": 20, "backend": "torch" if backend == "sdpa" else "triton"},
        layer_kernels=load_layer_kernels(kernel_config=kernels, instance=False),
        node_coords=icosahedron_coords(RESOLUTION).float(),
    )
    layer.gradient_checkpointing = True
    return layer


def _assert_close(actual, expected, tolerance, name, scale_to_largest=False):
    atol = tolerance["atol"] * (max(1.0, expected.abs().max().item()) if scale_to_largest else 1.0)
    torch.testing.assert_close(actual, expected, rtol=tolerance["rtol"], atol=atol, msg=lambda m: f"{name}: {m}")


def _split_matches_one_gpu_rank(rank, world_size, device, group, radius, conditional, split):
    if device.type == "cpu":
        dtype, backend, tolerance = torch.float64, "sdpa", dict(rtol=1e-10, atol=1e-10)
    else:
        dtype, backend, tolerance = torch.float32, "triton", dict(rtol=1e-4, atol=1e-5)

    torch.manual_seed(0)
    one_gpu = _build(radius, backend, conditional)
    split_layer = _build(radius, backend, conditional)
    split_layer.load_state_dict(one_gpu.state_dict())
    one_gpu, split_layer = one_gpu.to(device, dtype), split_layer.to(device, dtype)
    num_points = 10 * 4**RESOLUTION + 2

    generator = torch.Generator().manual_seed(1)

    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(device, dtype)

    full = {"x": rand(num_points, NUM_CHANNELS)}
    if conditional:
        full["cond"] = rand(num_points, COND_CHANNELS)
    upstream = rand(num_points, NUM_CHANNELS)

    def run(layer, inputs, sizes, group):
        kwargs = {"cond": inputs["cond"]} if "cond" in inputs else {}
        return layer(inputs["x"], 1, GraphShardInfo(nodes=sizes), model_comm_group=group, **kwargs)

    ref_inputs = {name: value.clone().requires_grad_() for name, value in full.items()}
    ref_out = run(one_gpu, ref_inputs, [num_points], None)
    (ref_out * upstream).sum().backward()

    sizes = split_points(num_points, world_size, split)
    own = slice(sum(sizes[:rank]), sum(sizes[: rank + 1]))
    split_inputs = {name: value[own].clone().requires_grad_() for name, value in full.items()}
    out = run(split_layer, split_inputs, sizes, group)
    _assert_close(out, ref_out[own].detach(), tolerance, "output")
    (out * upstream[own]).sum().backward()
    for name, value in split_inputs.items():
        _assert_close(value.grad, ref_inputs[name].grad[own], tolerance, f"{name} grad")

    # A parameter gradient is a sum over all points, added up in another order when split, so its
    # rounding error scales with its largest entries rather than with each entry.
    ref_params = dict(one_gpu.named_parameters())
    for name, param in split_layer.named_parameters():
        assert param.grad is not None, name
        grad = param.grad.clone()
        dist.all_reduce(grad, group=group)
        _assert_close(grad, ref_params[name].grad, tolerance, name, scale_to_largest=True)


@pytest.mark.distributed
@pytest.mark.parametrize("variant", VARIANTS)
def test_split_across_gpus_matches_one_gpu(variant: str, distributed_backend: str, distributed_world_size: int) -> None:
    _run_distributed_test(
        _split_matches_one_gpu_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        **VARIANTS[variant],
    )
