# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A multiscale loss on a grid split across processes matches a single process.

With checkpoint_scales the all-to-all exchanges run again in the backward pass, so this also checks
that they line up across processes.
"""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import HeteroData

from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.training.losses.kcrps import CRPS
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.utils.enums import TensorDim
from tests.unit.grids import octahedral_graph

WORLD_SIZE = 2
SIGMAS = (0.08, 0.04)  # widest first
NUM_MEMBERS = 3
# One variable on two processes leaves one process without channels.
VARIABLE_COUNTS = (5, 1)


def _loss(graph: HeteroData, num_variables: int, checkpoint_scales: bool) -> MultiscaleLossWrapper:
    loss = MultiscaleLossWrapper(
        per_scale_loss=CRPS(alpha=0.95, backend="naive"),
        weights=[1.0, 2.0, 3.0],
        multiscale_config={"num_nearest_neighbours": 9, "sigmas": list(SIGMAS)},
        sparse_projector_backend="torch",
        graph_data=graph,
        data_node_name="data",
        checkpoint_scales=checkpoint_scales,
    )
    generator = torch.Generator().manual_seed(1)
    loss.add_scaler(TensorDim.GRID, torch.rand(graph["data"].num_nodes, generator=generator) + 0.5, name="node_weights")
    loss.add_scaler(TensorDim.VARIABLE, torch.rand(num_variables, generator=generator) + 0.5, name="variables")
    return loss


def _check_against_one_process(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )
    try:
        model_group = dist.new_group(list(range(WORLD_SIZE)))
        graph = octahedral_graph(4)
        num_points = graph["data"].num_nodes
        grid_shard_sizes = get_balanced_partition_sizes(num_points, WORLD_SIZE)
        start = sum(grid_shard_sizes[:rank])
        grid_slice = slice(start, start + grid_shard_sizes[rank])

        for num_variables in VARIABLE_COUNTS:
            generator = torch.Generator().manual_seed(0)
            prediction = torch.randn(1, 1, NUM_MEMBERS, num_points, num_variables, generator=generator)
            target = torch.randn(1, 1, 1, num_points, num_variables, generator=generator)

            # The reference: one process scoring the whole grid.
            reference_prediction = prediction.clone().requires_grad_()
            reference_loss = _loss(graph, num_variables, checkpoint_scales=False)(reference_prediction, target)
            reference_loss.backward()

            for checkpoint_scales in [False, True]:
                own = prediction[:, :, :, grid_slice].clone().requires_grad_()
                value = _loss(graph, num_variables, checkpoint_scales)(
                    own,
                    target[:, :, :, grid_slice],
                    group=model_group,
                    grid_shard_slice=grid_slice,
                    grid_shard_sizes=grid_shard_sizes,
                    grid_dim=TensorDim.GRID,
                )
                value.backward()

                context = f"rank={rank}, num_variables={num_variables}, checkpoint_scales={checkpoint_scales}"
                torch.testing.assert_close(value, reference_loss, rtol=1e-5, atol=1e-6, msg=context)
                torch.testing.assert_close(
                    own.grad,
                    reference_prediction.grad[:, :, :, grid_slice],
                    rtol=1e-5,
                    atol=1e-6,
                    msg=context,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
def test_checkpointed_scales_on_a_split_grid_match_one_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loss and the gradient of each process's grid points match one process."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    mp.spawn(_check_against_one_process, args=(str(tmp_path / "rendezvous"),), nprocs=WORLD_SIZE, join=True)
