# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from anemoi.training.losses.scaler_tensor import LossScaler
from anemoi.training.schemas.training import BaseTrainingSchema
from anemoi.training.schemas.training import LossGradientScalingSchema
from anemoi.training.train.methods.base import BaseTrainingModule

NORM_DIMS = (2, 3)  # ensemble, grid


def scaled_grad(grad: torch.Tensor, scaler: LossScaler | None = None, group: object = None) -> torch.Tensor:
    x = torch.zeros(grad.shape, dtype=grad.dtype, requires_grad=True)
    out = (scaler or LossScaler())(x, group=group)
    out.backward(grad)
    return x.grad


def var_norms(grad: torch.Tensor) -> torch.Tensor:
    return grad.float().square().sum(dim=NORM_DIMS).sqrt()


@pytest.fixture
def grad() -> torch.Tensor:
    torch.manual_seed(0)
    # (batch, time, ens, grid, var), with very different scales per variable
    return torch.randn(2, 3, 4, 50, 6) * torch.logspace(-6, 0, 6)


def test_forward_is_identity(grad: torch.Tensor) -> None:
    x = torch.randn_like(grad)
    assert torch.equal(LossScaler()(x), x)


def test_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="5D"):
        LossScaler()(torch.zeros(2, 3, 4, requires_grad=True))


def test_equalises_variable_norms(grad: torch.Tensor) -> None:
    norms = var_norms(scaled_grad(grad))
    torch.testing.assert_close(norms, norms[..., :1].expand_as(norms))


@pytest.mark.parametrize("alpha", [1e-8, 1.0, 1e6])
def test_positively_homogeneous(grad: torch.Tensor, alpha: float) -> None:
    torch.testing.assert_close(scaled_grad(alpha * grad), alpha * scaled_grad(grad))


def test_zero_gradient_variable_is_inactive(grad: torch.Tensor) -> None:
    grad = grad.clone()
    grad[..., 3] = 0.0
    out = scaled_grad(grad)
    assert torch.isfinite(out).all()
    assert (out[..., 3] == 0).all()
    keep = [0, 1, 2, 4, 5]
    torch.testing.assert_close(out[..., keep], scaled_grad(grad[..., keep].contiguous()))


def test_all_zero_row_stays_zero(grad: torch.Tensor) -> None:
    grad = grad.clone()
    grad[0, 1] = 0.0
    out = scaled_grad(grad)
    assert torch.isfinite(out).all()
    assert (out[0, 1] == 0).all()


def test_power_zero_is_identity(grad: torch.Tensor) -> None:
    torch.testing.assert_close(scaled_grad(grad, LossScaler(power=0.0)), grad)


def test_max_gain_bounds_weight_ratio(grad: torch.Tensor) -> None:
    out = scaled_grad(grad, LossScaler(max_gain=10.0))
    weight = var_norms(out) / var_norms(grad)
    ratio = weight.amax(dim=-1) / weight.amin(dim=-1)
    assert (ratio <= 10.0 * (1 + 1e-5)).all()
    assert (ratio >= 10.0 * (1 - 1e-5)).all()  # the 1e6 spread in the fixture saturates the cap


def test_bf16_gradient(grad: torch.Tensor) -> None:
    out = scaled_grad(grad.bfloat16())
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), scaled_grad(grad.bfloat16().float()), rtol=1e-2, atol=1e-3)


def test_weights_shared_across_ensemble(grad: torch.Tensor) -> None:
    ratio = scaled_grad(grad) / grad
    torch.testing.assert_close(ratio, ratio[:, :, :1, :1].expand_as(ratio))


def test_sharded_grid_matches_full_grid(grad: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> None:
    shards = grad[..., :20, :].contiguous(), grad[..., 20:, :].contiguous()
    full = scaled_grad(grad)
    shard_group = object()

    def fake_world_size(group: object) -> int:
        assert group is shard_group
        return 2

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_world_size)
    for rank, shard in enumerate(shards):
        other = shards[1 - rank]

        def fake_all_reduce(tensor: torch.Tensor, group: object, other: torch.Tensor = other) -> None:
            assert group is shard_group
            tensor += other.float().square().sum(dim=NORM_DIMS, keepdim=True)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        expected = full[..., :20, :] if rank == 0 else full[..., 20:, :]
        torch.testing.assert_close(scaled_grad(shard, group=shard_group), expected)


def test_scaler_has_no_state() -> None:
    assert LossScaler(power=0.5, max_gain=100.0).state_dict() == {}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, None),
        (False, None),
        (True, (1.0, None)),
        (LossGradientScalingSchema(), None),
        (LossGradientScalingSchema(enabled=True, power=0.5, max_gain=100.0), (0.5, 100.0)),
        (OmegaConf.create({"enabled": True, "max_gain": 10.0}), (1.0, 10.0)),
        (OmegaConf.create({"enabled": False}), None),
    ],
)
def test_from_config(config: object, expected: tuple[float, float | None] | None) -> None:
    scaler = LossScaler.from_config(config)
    if expected is None:
        assert scaler is None
    else:
        assert (scaler.power, scaler.max_gain) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, LossGradientScalingSchema()),
        (True, LossGradientScalingSchema(enabled=True)),
        (
            {"enabled": True, "power": 0.5, "max_gain": 100},
            LossGradientScalingSchema(enabled=True, power=0.5, max_gain=100),
        ),
    ],
)
def test_schema_accepts_legacy_bool(value: object, expected: LossGradientScalingSchema) -> None:
    coerced = BaseTrainingSchema.coerce_loss_gradient_scaling(value)
    assert LossGradientScalingSchema.model_validate(coerced) == expected


def test_schema_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        LossGradientScalingSchema(enabled=True, max_gain=0.5)
    with pytest.raises(ValueError, match="greater than or equal"):
        LossGradientScalingSchema(enabled=True, power=-1.0)


class WeightedSquaredLoss(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **_kwargs) -> torch.Tensor:
        return ((pred - target).square() * self.weights).sum()


class GroupSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.groups = []

    def forward(self, x: torch.Tensor, group: object = None) -> torch.Tensor:
        self.groups.append(group)
        return x


def _module(scaler: nn.Module | None, model_comm_group: object = None) -> SimpleNamespace:
    loss = WeightedSquaredLoss(torch.tensor([1.0, 3.0, 10.0, 1.0]))
    return SimpleNamespace(
        loss={"data": loss},
        model_comm_group=model_comm_group,
        loss_gradient_scaler=scaler,
        grid_shard_sizes={"data": None},
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_compute_loss_applies_scaler(enabled: bool) -> None:
    torch.manual_seed(0)
    module = _module(LossScaler() if enabled else None)
    pred = torch.randn(1, 1, 1, 30, 4, requires_grad=True)
    target = torch.zeros_like(pred)
    BaseTrainingModule._compute_loss(module, pred, target, dataset_name="data").backward()
    norms = var_norms(pred.grad)[0, 0]
    equal = torch.allclose(norms, norms[:1].expand_as(norms))
    assert equal == enabled


@pytest.mark.parametrize("sharded", [False, True])
def test_compute_loss_passes_group_only_when_sharded(sharded: bool) -> None:
    group = object()
    spy = GroupSpy()
    module = _module(spy, model_comm_group=group)
    pred = torch.randn(1, 1, 1, 30, 4, requires_grad=True)
    BaseTrainingModule._compute_loss(
        module,
        pred,
        torch.zeros_like(pred),
        grid_shard_slice=slice(0, 30) if sharded else None,
        dataset_name="data",
    )
    assert spy.groups == [group if sharded else None]
