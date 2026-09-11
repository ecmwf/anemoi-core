# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

from __future__ import annotations

import math

import torch

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.transport.objectives import EDMDiffusionModelObjective
from anemoi.models.transport.settings import EdmSettings


class _Model:
    edm = EdmSettings(sigma_data=1.0)

    def __init__(self):
        self.scaled_noised = None

    def _assert_condition_shapes(self, sigma):
        return 2, 1

    def _forward_transport_network(self, _x, scaled_noised, _c_noise, **_kwargs):
        self.scaled_noised = scaled_noised
        return scaled_noised.with_data(
            {"obs": [torch.zeros_like(sample) for sample in scaled_noised.data["obs"]]},
        )


def test_edm_model_preconditioning_handles_sparse_targets() -> None:
    model = _Model()
    noised = Batch(
        data={"obs": [torch.ones(2, 1), torch.full((3, 1), 2.0)]},
        coordinates={"obs": [torch.zeros(2, 2), torch.zeros(3, 2)]},
        metadata={"obs": {"boundaries": [(slice(0, 2),), (slice(0, 3),)]}},
        layouts={"obs": TensorLayout(grid=0, variables=1, time_in_grid=True)},
        variables={"obs": ["x"]},
        statistics={"obs": {}},
    )
    sigma = {"obs": torch.ones(2, 1, 1, 1, 1)}

    out = EDMDiffusionModelObjective().forward(model, noised.with_data({}), noised, sigma)

    torch.testing.assert_close(model.scaled_noised.data["obs"][0], torch.full((2, 1), 1.0 / math.sqrt(2.0)))
    torch.testing.assert_close(model.scaled_noised.data["obs"][1], torch.full((3, 1), math.sqrt(2.0)))
    torch.testing.assert_close(out.data["obs"][0], torch.full((2, 1), 0.5))
    torch.testing.assert_close(out.data["obs"][1], torch.full((3, 1), 1.0))
