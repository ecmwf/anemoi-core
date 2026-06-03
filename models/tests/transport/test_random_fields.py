# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

import math

import torch

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
