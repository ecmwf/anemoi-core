# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A whole encoder-processor-decoder model with neighbourhood attention gives the same results split across GPUs.

Every rank builds the same model (encoder, processor and decoder with neighbourhood attention,
rotary embeddings and gradient checkpointing) and runs it once on its own over the whole grids
and once with the points split across the ranks: either the input already split, as training
does with the batch kept split, or the whole input given to every rank and split by the model.
Outputs, input gradients and the parameter gradients summed over the ranks must match. On CPU
(gloo) in float64 with the dense-mask backend; on GPU (nccl) in float32 with the Triton kernels.

Run with ``pytest --distributed [--distributed-world-size N] [--distributed-backend nccl]``.
"""

import math

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.models import AnemoiModelEncProcDec
from tests.distributed._distributed_runner import _run_distributed_test

NUM_CHANNELS = 32
COMPLETE_ON_EVERY_RANK = ("trainable", "no_gradscaling")

CASES = {
    # name: (grid family, data grid, hidden grid)
    "octahedral O32 / O16": ("octahedral", ReducedGrid.octahedral(32), ReducedGrid.octahedral(16)),
    "healpix H16 / H8": ("healpix", ReducedGrid.healpix(16), ReducedGrid.healpix(8)),
}


def _grid_coords(grid: ReducedGrid) -> torch.Tensor:
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    shifts = torch.tensor(grid.shifts, dtype=torch.float64)[rows]
    lon = 2 * math.pi * (positions + shifts / 2) / lengths
    return torch.stack([lat, lon], dim=1).float()


def _transformer(family: str, kernel_size, num_bands: int, backend: str) -> dict:
    return {
        "num_channels": NUM_CHANNELS,
        "num_chunks": 1,
        "num_heads": 2,  # head dim 16, the smallest the Triton kernels take
        "mlp_hidden_ratio": 2,
        "window_size": None,
        "dropout_p": 0.0,
        "attention_implementation": "neighbourhood",
        "neighbourhood": {"grid": family, "kernel_size": list(kernel_size), "backend": backend, "num_bands": num_bands},
        "rotary_embeddings": {"max_frequency": 30.0, "backend": "torch" if backend == "sdpa" else "triton"},
        "softcap": 0.0,
        "cpu_offload": False,
        "gradient_checkpointing": True,
        "layer_kernels": {},
    }


def _build(case: str, num_bands: int, backend: str) -> AnemoiModelEncProcDec:
    family, data, hidden = CASES[case]
    config = OmegaConf.create(
        {
            "model": {
                "model": {
                    "_target_": "anemoi.models.models.AnemoiModelEncProcDec",
                    "hidden_nodes_name": "hidden",
                    "latent_skip": True,
                },
                "node_trainable_parameters": {"data": 4, "hidden": 4},
                "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
                "processor": {
                    "_target_": "anemoi.models.layers.processor.TransformerProcessor",
                    "num_layers": 2,
                    "qk_norm": True,
                    **_transformer(family, (7, 13) if family == "octahedral" else (5, 5), num_bands, backend),
                },
                "encoders": {
                    "0": {
                        "source_datasets": ["data"],
                        "dataset_fusing_strategy": "not_supported",
                        "mapper": {
                            "_target_": "anemoi.models.layers.mapper.TransformerForwardMapper",
                            "qk_norm": True,
                            **_transformer(family, (5, 5), num_bands, backend),
                        },
                    },
                },
                "decoders": {
                    "0": {
                        "target_datasets": ["data"],
                        "target_node_features": ["encoded_data"],
                        "mapper": {
                            "_target_": "anemoi.models.layers.mapper.TransformerBackwardMapper",
                            "qk_norm": True,
                            **_transformer(family, (3, 5), num_bands, backend),
                        },
                    },
                },
                "residual": {
                    "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}},
                },
                "bounding": {"datasets": {"data": []}},
                "output_mask": {"datasets": {"data": {"_target_": "anemoi.training.utils.masks.NoOutputMask"}}},
            },
        }
    )
    graph = HeteroData()
    for name, grid in (("data", data), ("hidden", hidden)):
        graph[name].x = _grid_coords(grid)
        graph[name].num_nodes = grid.num_points
    data_config = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": []})
    data_indices = {"data": IndexCollection(data_config, {"prog0": 0, "prog1": 1, "force": 2, "diag": 3})}
    return AnemoiModelEncProcDec(
        model_config=config,
        data_indices=data_indices,
        statistics={"data": None},
        n_step_input=2,
        n_step_output=1,
        graph_data=graph,
    )


def _model_matches_one_gpu_rank(rank, world_size, device, group, case, num_bands, input_split):
    if device.type == "cpu":
        dtype, backend, rtol, atol = torch.float64, "sdpa", 1e-10, 1e-10
    else:
        dtype, backend, rtol, atol = torch.float32, "triton", 1e-4, 1e-5
    _, data, _ = CASES[case]
    num_points = data.num_points

    torch.manual_seed(0)
    one_gpu = _build(case, 1, backend)
    split_model = _build(case, num_bands, backend)
    split_model.load_state_dict(one_gpu.state_dict())
    one_gpu, split_model = one_gpu.to(device, dtype), split_model.to(device, dtype)

    generator = torch.Generator().manual_seed(1)
    num_input_vars = 3
    x_full = torch.randn(1, 2, 1, num_points, num_input_vars, generator=generator, dtype=torch.float64).to(
        device, dtype
    )

    # The reference: this rank alone, over the whole grids.
    x_ref = x_full.clone().requires_grad_()
    ref_out = one_gpu({"data": x_ref})["data"]
    upstream = torch.randn(ref_out.shape, generator=generator, dtype=torch.float64).to(device, dtype)
    (ref_out * upstream).sum().backward()

    sizes = get_balanced_partition_sizes(num_points, world_size)
    own = slice(sum(sizes[:rank]), sum(sizes[: rank + 1]))
    if input_split:
        # The batch kept split: each rank passes in and gets back its own points.
        x = x_full[..., own, :].clone().requires_grad_()
        out = split_model({"data": x}, model_comm_group=group, grid_shard_sizes={"data": sizes})["data"]
        torch.testing.assert_close(
            out, ref_out[..., own, :].detach(), rtol=rtol, atol=atol, msg=lambda m: f"output: {m}"
        )
        (out * upstream[..., own, :]).sum().backward()
        torch.testing.assert_close(
            x.grad, x_ref.grad[..., own, :], rtol=rtol, atol=atol, msg=lambda m: f"input grad: {m}"
        )
    else:
        # Every rank passes in the whole input and gets back the whole output; the model splits inside.
        x = x_full.clone().requires_grad_()
        out = split_model({"data": x}, model_comm_group=group)["data"]
        torch.testing.assert_close(out, ref_out.detach(), rtol=rtol, atol=atol, msg=lambda m: f"output: {m}")
        (out * upstream).sum().backward()

    ref_params = dict(one_gpu.named_parameters())
    with_grad = 0
    for name, param in split_model.named_parameters():
        assert (param.grad is None) == (ref_params[name].grad is None), name
        if param.grad is None:
            continue
        grad = param.grad.clone()
        # Most parameters get from each rank the gradient of its own points, which add up to the
        # whole. Trainable node features are split with shard_tensor, whose backward gathers, so
        # every rank already holds their whole gradient; training leaves these out of its
        # gradient scaling for the same reason (register_gradient_scaling_hooks).
        if not any(part in name for part in COMPLETE_ON_EVERY_RANK):
            dist.all_reduce(grad, group=group)
        expected = ref_params[name].grad
        # A parameter gradient sums over all points, so its rounding error scales with its largest entries.
        scale = max(1.0, expected.abs().max().item())
        torch.testing.assert_close(grad, expected, rtol=rtol, atol=atol * scale, msg=lambda m, n=name: f"{n}: {m}")
        with_grad += 1
    assert with_grad > 0.9 * len(ref_params)


@pytest.mark.distributed
@pytest.mark.parametrize("input_split", [True, False], ids=["input split", "whole input"])
@pytest.mark.parametrize("num_bands", [1, 4])
@pytest.mark.parametrize("case", CASES)
def test_split_model_matches_one_gpu(
    case: str, num_bands: int, input_split: bool, distributed_backend: str, distributed_world_size: int
) -> None:
    _run_distributed_test(
        _model_matches_one_gpu_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        case=case,
        num_bands=num_bands,
        input_split=input_split,
    )
