# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

import math

import torch

from anemoi.models.data.layout import TensorLayout
from anemoi.models.data.sources import TabularSource
from anemoi.models.transport import random_fields


def test_randn_with_grid_sharding_creates_full_grid_before_sharding(monkeypatch) -> None:
    calls = {}

    def fake_randn(shape, device=None, dtype=None):
        calls["shape"] = shape
        values = torch.arange(math.prod(shape), device=device, dtype=dtype)
        return values.reshape(shape)

    def fake_shard_tensor(input_, dim, sizes, mgroup):
        calls["sizes"] = sizes
        calls["mgroup"] = mgroup
        return input_.narrow(dim, 0, sizes[0])

    monkeypatch.setattr(torch, "randn", fake_randn)
    monkeypatch.setattr(random_fields, "shard_tensor", fake_shard_tensor)

    model_comm_group = object()
    noise = random_fields.randn_with_grid_sharding(
        (1, 2, 1, 3, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_comm_group=model_comm_group,
        grid_shard_sizes=[3, 5],
    )

    expected_full = torch.arange(1 * 2 * 1 * 8 * 4, dtype=torch.float32).reshape(1, 2, 1, 8, 4)
    torch.testing.assert_close(noise, expected_full.narrow(-2, 0, 3))
    assert calls["shape"] == (1, 2, 1, 8, 4)
    assert calls["sizes"] == [3, 5]
    assert calls["mgroup"] is model_comm_group


def _sharded_obs_source(local_nodes: int, rank_sizes: list[int]) -> TabularSource:
    """A one-sample, one-window observation source holding ``local_nodes`` of the model group's points."""
    return TabularSource(
        name="obs",
        data=[torch.zeros(1, local_nodes, 2)],
        coordinates=[torch.zeros(local_nodes, 2)],
        variables=["a", "b"],
        statistics={},
        layout=TensorLayout(ensemble=0, grid=1, variables=2),
        boundaries=[(slice(0, local_nodes),)],
        timedeltas=[torch.zeros(local_nodes)],
        shard_sizes=[[rank_sizes]],
    )


def test_sharded_tabular_noise_differs_across_ranks_and_keeps_the_random_state_in_step(monkeypatch) -> None:
    """Each rank keeps its own rows of noise drawn for all points, so ranks never repeat values."""
    rank_sizes = [2, 3]
    starts = [0, 2]
    noises, next_draws = [], []
    for rank in (0, 1):
        monkeypatch.setattr(
            random_fields,
            "shard_tensor",
            lambda input_, dim, sizes, mgroup, rank=rank: input_.narrow(dim, starts[rank], sizes[rank]),
        )
        torch.manual_seed(0)
        noises.append(_sharded_obs_source(rank_sizes[rank], rank_sizes).randn_like(model_comm_group=object()))
        next_draws.append(torch.randn(1))

    torch.manual_seed(0)
    full = torch.randn(1, 5, 2)
    torch.testing.assert_close(noises[0].data[0], full[:, :2])
    torch.testing.assert_close(noises[1].data[0], full[:, 2:])
    # Both ranks consumed the same amount of randomness, so later draws (noise levels, times) agree.
    torch.testing.assert_close(next_draws[0], next_draws[1])
