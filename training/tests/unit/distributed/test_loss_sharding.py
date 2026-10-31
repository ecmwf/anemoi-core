# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare grid-sharded smoothing and graph losses with a single-process reference on CPU."""

import math
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.training.losses import CRPS
from anemoi.training.losses import GraphEdgeCRPSLoss
from anemoi.training.losses import GraphEdgeEnergyScoreLoss
from anemoi.training.losses import GraphEnergyScoreLoss
from anemoi.training.losses import GraphVariogramScoreLoss
from anemoi.training.losses import MultiscaleLossWrapper
from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.enums import TensorDim

# 13 grid points do not split evenly over 2 or 3 ranks.
N_GRID = 13
N_TIME = 2
N_MEMBERS = 3
GRID_AXIS = 3
LOSS_GRAPH = {"edges_name": ["data", "to", "data"], "edge_weight_attribute": "weight", "row_normalize": True}
SMOOTHING = {"num_scales": 2, "base_num_nearest_neighbours": 4, "base_sigma": 0.05}
# Fewer, as many as, and more variables than the 2 or 3 ranks, so some ranks get no variable
# when the variables are shared out for smoothing or for the graph scores.
VARIABLE_COUNTS = (1, 2, 3, 4)


def _graph() -> HeteroData:
    """Random points on the sphere, each joined to itself and its three nearest neighbours."""
    generator = torch.Generator().manual_seed(0)
    lat = torch.asin(2 * torch.rand(N_GRID, generator=generator) - 1)
    lon = 2 * math.pi * torch.rand(N_GRID, generator=generator)
    graph = HeteroData()
    graph["data"].x = torch.stack([lat, lon], dim=-1)
    graph["data"].num_nodes = N_GRID
    xyz = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)
    nearest = torch.cdist(xyz, xyz).topk(4, largest=False).indices
    destination = torch.arange(N_GRID).repeat_interleave(4)
    graph["data", "to", "data"].edge_index = torch.stack([nearest.reshape(-1), destination])
    graph["data", "to", "data"].weight = torch.rand(destination.numel(), generator=generator) + 0.5
    return graph


def _loss_builders(graph: HeteroData) -> dict[str, Callable[[bool], BaseLoss]]:
    return {
        "multiscale CRPS": lambda ignore_nans: MultiscaleLossWrapper(
            per_scale_loss=CRPS(ignore_nans=ignore_nans),
            weights=[1.0, 0.5, 0.25],
            multiscale_config=SMOOTHING,
            graph_data=graph,
            ignore_nans=ignore_nans,
        ),
        "graph energy": lambda ignore_nans: GraphEnergyScoreLoss(
            loss_graph=LOSS_GRAPH,
            graph_data=graph,
            ignore_nans=ignore_nans,
        ),
        "graph edge energy": lambda ignore_nans: GraphEdgeEnergyScoreLoss(
            loss_graph=LOSS_GRAPH,
            graph_data=graph,
            ignore_nans=ignore_nans,
        ),
        "graph edge CRPS": lambda ignore_nans: GraphEdgeCRPSLoss(
            loss_graph=LOSS_GRAPH,
            graph_data=graph,
            ignore_nans=ignore_nans,
        ),
        "graph variogram": lambda ignore_nans: GraphVariogramScoreLoss(
            loss_graph=LOSS_GRAPH,
            graph_data=graph,
            ignore_nans=ignore_nans,
        ),
    }


def _inputs(n_variables: int, with_nans: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(n_variables)
    prediction = torch.randn(1, N_TIME, N_MEMBERS, N_GRID, n_variables, generator=generator)
    target = torch.randn(1, N_TIME, 1, N_GRID, n_variables, generator=generator)
    if with_nans:
        target[:, :, :, ::4, 0] = float("nan")
    grid_weights = torch.rand(N_GRID, generator=generator) + 0.5
    return prediction, target, grid_weights


def _loss_and_gradient(
    loss: BaseLoss,
    prediction: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, slice]:
    """Evaluate the loss on this rank's part of the grid, or on the whole grid without a group."""
    if group is None:
        local, kwargs = slice(None), {}
    else:
        shard_sizes = get_shard_sizes(prediction, GRID_AXIS, model_comm_group=group)
        start = sum(shard_sizes[: group.rank()])
        local = slice(start, start + shard_sizes[group.rank()])
        kwargs = {"grid_shard_slice": local, "group": group}
        if loss.needs_shard_layout_info:
            kwargs |= {"grid_shard_sizes": shard_sizes, "grid_dim": -2}
    local_prediction = prediction[:, :, :, local].clone().requires_grad_()
    local_target = target[:, :, :, local].clone()
    value = loss(local_prediction, local_target, **kwargs)
    value.backward()
    return value.detach(), local_prediction.grad, local


def _check_sharded_losses(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        group = dist.new_group(list(range(world_size)))
        graph = _graph()
        mismatches = []
        # Every rank runs every case before any check fails, so no rank is left waiting in a
        # collective that the others never reach.
        for name, build in _loss_builders(graph).items():
            for n_variables in VARIABLE_COUNTS:
                for with_nans in (False, True):
                    case = f"{name}, {n_variables} variable(s), NaN targets: {with_nans}"
                    prediction, target, grid_weights = _inputs(n_variables, with_nans)
                    values = {}
                    for sharded in (False, True):
                        loss = build(with_nans)
                        loss.add_scaler(TensorDim.GRID, grid_weights, name="node_weights")
                        values[sharded] = _loss_and_gradient(loss, prediction, target, group if sharded else None)
                    (expected, expected_grad, _), (actual, actual_grad, local) = values[False], values[True]
                    if not torch.isfinite(actual).all():
                        mismatches.append(f"{case}: loss is not finite ({actual})")
                    if not torch.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                        mismatches.append(f"{case}: loss {actual} differs from {expected}")
                    if not torch.allclose(actual_grad, expected_grad[:, :, :, local], rtol=1e-10, atol=1e-12):
                        mismatches.append(f"{case}: gradient differs")
        assert not mismatches, f"rank {rank} of {world_size}:\n" + "\n".join(mismatches)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("world_size", [2, 3])
def test_grid_sharded_losses_match_single_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    """Losses on a grid split over ranks give the single-process loss and gradients.

    The grid does not split evenly, and some cases have fewer variables than ranks, so some ranks
    hold no variable while the variables are shared out.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(_check_sharded_losses, args=(world_size, str(tmp_path / "rendezvous")), nprocs=world_size, join=True)
