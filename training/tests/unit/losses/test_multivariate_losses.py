# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from anemoi.training.losses import EnergyScore
from anemoi.training.losses import GroupedEnergyScore
from anemoi.training.losses import GroupedMultivariateKernelCRPS
from anemoi.training.losses import MultivariateKernelCRPS
from anemoi.training.losses import VariogramScore
from anemoi.training.utils.enums import TensorDim


class _Dummy:
    pass


def _make_data_indices(name_to_index: dict[str, int]) -> Any:
    di = _Dummy()
    di.model = _Dummy()
    di.model.output = _Dummy()
    di.model.output.name_to_index = name_to_index
    return di


def test_multivariate_kcrps_runs_and_returns_scalar() -> None:
    torch.manual_seed(0)
    bs, ens, grid, var = 2, 3, 7, 4
    pred = torch.randn(bs, ens, grid, var)
    target = torch.randn(bs, grid, var)

    loss = MultivariateKernelCRPS()(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar
    assert torch.isfinite(loss)


def test_multivariate_kcrps_scaler_indices_matches_manual_subsetting() -> None:
    torch.manual_seed(0)
    bs, ens, grid, var = 2, 3, 7, 5
    pred = torch.randn(bs, ens, grid, var)
    target = torch.randn(bs, grid, var)
    idxs = torch.tensor([0, 3, 4], dtype=torch.long)

    loss_fn = MultivariateKernelCRPS()
    a = loss_fn(pred, target, scaler_indices=[..., idxs])
    b = loss_fn(pred[..., idxs], target[..., idxs])
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_energy_score_matches_multivariate_kcrps_p2() -> None:
    torch.manual_seed(0)
    bs, ens, grid, var = 2, 3, 7, 4
    pred = torch.randn(bs, ens, grid, var)
    target = torch.randn(bs, grid, var)

    a = EnergyScore(beta=1.0, fair=True)(pred, target)
    b = MultivariateKernelCRPS(beta=1.0, fair=True, p_norm=2.0)(pred, target)
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_grouped_multivariate_kcrps_requires_data_indices() -> None:
    torch.manual_seed(0)
    bs, ens, grid, var = 2, 3, 7, 4
    pred = torch.randn(bs, ens, grid, var)
    target = torch.randn(bs, grid, var)

    gm = GroupedMultivariateKernelCRPS(patch_method="group_by_variable")
    with pytest.raises(ValueError, match="requires data indices"):
        _ = gm(pred, target)


@pytest.mark.parametrize("loss_cls", [GroupedMultivariateKernelCRPS, GroupedEnergyScore])
def test_grouped_feature_grouping_matches_manual_sum(
    loss_cls: type[GroupedMultivariateKernelCRPS] | type[GroupedEnergyScore],
) -> None:
    torch.manual_seed(0)
    bs, ens, grid, var = 2, 3, 7, 4
    pred = torch.randn(bs, ens, grid, var)
    target = torch.randn(bs, grid, var)

    # Two groups: t_* and q_* (each of length 2)
    name_to_index = {"t_500": 0, "t_850": 1, "q_500": 2, "q_850": 3}
    di = _make_data_indices(name_to_index)

    grouped = (
        loss_cls(patch_method="group_by_variable", beta=1.0, fair=True)
        if loss_cls is GroupedEnergyScore
        else loss_cls(
            patch_method="group_by_variable",
            beta=1.0,
            fair=True,
            p_norm=2.0,
        )
    )
    grouped.set_data_indices(di)

    # manual sum of group losses should match grouped implementation (linearity)
    base = (
        EnergyScore(beta=1.0, fair=True)
        if loss_cls is GroupedEnergyScore
        else MultivariateKernelCRPS(beta=1.0, fair=True, p_norm=2.0)
    )

    idx_t = torch.tensor([0, 1], dtype=torch.long)
    idx_q = torch.tensor([2, 3], dtype=torch.long)
    manual = base(pred[..., idx_t], target[..., idx_t]) + base(pred[..., idx_q], target[..., idx_q])
    got = grouped(pred, target)
    assert torch.allclose(got, manual, rtol=1e-6, atol=1e-6)


def test_variogram_known_value_no_scalers() -> None:
    # bs=1, ens=1, grid=1: reduce is identity
    pred = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # (1,1,1,3)
    target = torch.tensor([[[1.0, 2.0, 4.0]]])  # (1,1,3)

    beta1, beta2 = 1.0, 2.0
    v = VariogramScore(beta1=beta1, beta2=beta2)
    got = v(pred, target)

    # obs pairwise diffs: |1-2|=1, |1-4|=3, |2-4|=2
    # score = average_{i!=j} |diff_obs - diff_fcst|^2 ; diff_fcst=0
    # matrix off-diagonal has values: 1^2, 3^2, 2^2 twice each
    expected = (2 * (1**2 + 3**2 + 2**2)) / (3 * (3 - 1))
    assert math.isclose(float(got), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_variogram_variable_scaler_affects_metric() -> None:
    # With a variable scaler applied inside the metric, the score should change.
    pred = torch.tensor([[[[0.0, 0.0]]]])  # (1,1,1,2)
    target = torch.tensor([[[1.0, 2.0]]])  # (1,1,2)

    v0 = VariogramScore(beta1=1.0, beta2=2.0)
    base = v0(pred, target)

    v1 = VariogramScore(beta1=1.0, beta2=2.0)
    # Apply VAR scaler: components scaled by w^(1/beta1)=w before pairwise diffs
    v1.add_scaler(TensorDim.VARIABLE.value, torch.tensor([2.0, 1.0]), name="variable")
    scaled = v1(pred, target)

    # Unscaled obs diff = |1-2| = 1 => score = 1^2 = 1
    # Scaled target = [2,2] => obs diff = |2-2| = 0 => score = 0
    assert float(base) > 0.0
    assert math.isclose(float(scaled), 0.0, rel_tol=1e-6, abs_tol=1e-6)
