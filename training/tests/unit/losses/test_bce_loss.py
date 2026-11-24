# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.models.data import Source
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import BinaryCrossEntropyLoss
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import TimeAggregateLossWrapper
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.schemas.training import BinaryCrossEntropyLossSchema
from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace

TARGET = "anemoi.training.losses.BinaryCrossEntropyLoss"
GRID = 12
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


def _src(data: torch.Tensor) -> Source:
    """Wrap a ``(bs, time, ens, grid, nvar)`` tensor in the source view the losses take."""
    return GriddedSource(
        name="data",
        values=data,
        variables=[f"variable_{index}" for index in range(data.shape[-1])],
        statistics={},
        coordinates=torch.zeros(data.shape[GRIDDED_LAYOUT.grid], 2, device=data.device),
        layout=GRIDDED_LAYOUT,
    )


def _loss(**kwargs) -> BinaryCrossEntropyLoss:
    """BCE whose grid reduction is a mean (uniform unit-sum node weights)."""
    loss = BinaryCrossEntropyLoss(**kwargs)
    loss.add_scaler(TensorDim.GRID, torch.ones(GRID) / GRID, name="node_weights")
    return loss


def _data(nvars: int = 1, bs: int = 2, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    pred = torch.rand(bs, 1, 1, GRID, nvars, generator=gen) * 0.98 + 0.01
    target = (torch.rand(bs, 1, 1, GRID, nvars, generator=gen) > 0.7).float()
    return pred, target


def test_is_functional_loss() -> None:
    assert isinstance(BinaryCrossEntropyLoss(), FunctionalLoss)


@pytest.mark.parametrize("nvars", [1, 3])
def test_plain_bce_matches_torch(nvars: int) -> None:
    pred, target = _data(nvars)
    out = _loss()(_src(pred), _src(target))
    torch.testing.assert_close(out, F.binary_cross_entropy(pred, target), rtol=1e-5, atol=1e-6)


def test_plain_bce_matches_torch_with_soft_targets() -> None:
    pred, _ = _data()
    target = torch.rand_like(pred)
    out = _loss()(_src(pred), _src(target))
    torch.testing.assert_close(out, F.binary_cross_entropy(pred, target), rtol=1e-5, atol=1e-6)


def test_unsquashed_and_sum_modes() -> None:
    pred, target = _data(nvars=3)
    loss = _loss()
    per_var = loss(_src(pred), _src(target), squash=False)
    assert per_var.shape == (3,)
    torch.testing.assert_close(loss(_src(pred), _src(target)), per_var.mean())
    torch.testing.assert_close(loss(_src(pred), _src(target), squash_mode="sum"), per_var.sum())
    for i in range(3):
        torch.testing.assert_close(
            per_var[i],
            F.binary_cross_entropy(pred[..., i], target[..., i]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_focal_formula_and_down_weights_easy_examples() -> None:
    gamma = 2.0
    pred = torch.tensor([0.9, 0.1, 0.1, 0.9]).reshape(1, 1, 1, 4, 1)
    target = torch.tensor([1.0, 1.0, 0.0, 0.0]).reshape(1, 1, 1, 4, 1)
    pointwise = BinaryCrossEntropyLoss(focal_gamma=gamma).calculate_difference(pred, target)
    plain = BinaryCrossEntropyLoss().calculate_difference(pred, target)

    expected = plain * torch.tensor([0.1, 0.9, 0.1, 0.9]).reshape_as(plain) ** gamma
    torch.testing.assert_close(pointwise, expected)
    # Easy (well predicted) points are down-weighted much more than hard ones.
    ratio = pointwise / plain
    assert ratio[0, 0, 0, 0, 0] < ratio[0, 0, 0, 1, 0]
    assert ratio[0, 0, 0, 2, 0] < ratio[0, 0, 0, 3, 0]
    assert torch.all(pointwise < plain)


def test_alpha_and_pos_weight() -> None:
    pred = torch.tensor([0.3, 0.3]).reshape(1, 1, 1, 2, 1)
    target = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 2, 1)
    plain = BinaryCrossEntropyLoss().calculate_difference(pred, target)

    with_alpha = BinaryCrossEntropyLoss(alpha=0.25).calculate_difference(pred, target)
    torch.testing.assert_close(with_alpha, plain * torch.tensor([0.25, 0.75]).reshape_as(plain))

    with_pos = BinaryCrossEntropyLoss(pos_weight=5.0).calculate_difference(pred, target)
    torch.testing.assert_close(with_pos, plain * torch.tensor([5.0, 1.0]).reshape_as(plain))

    both = BinaryCrossEntropyLoss(alpha=0.25, pos_weight=5.0).calculate_difference(pred, target)
    torch.testing.assert_close(both, plain * torch.tensor([1.25, 0.75]).reshape_as(plain))


@pytest.mark.parametrize("kwargs", [{"focal_gamma": -1.0}, {"alpha": 1.5}, {"pos_weight": 0.0}, {"eps": 0.0}])
def test_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be"):
        BinaryCrossEntropyLoss(**kwargs)


@pytest.mark.parametrize("ignore_nans", [True, False])
@pytest.mark.parametrize("focal_gamma", [0.0, 2.0])
def test_nan_targets_are_excluded_from_loss_and_normalisation(ignore_nans: bool, focal_gamma: float) -> None:
    pred, target = _data(bs=3)
    nan_target = target.clone()
    nan_target[0, ..., :4, :] = torch.nan  # a block of one sample
    nan_target[2, ..., 5, :] = torch.nan  # a single point of another sample
    valid = ~torch.isnan(nan_target)

    loss = _loss(focal_gamma=focal_gamma, ignore_nans=ignore_nans)
    out = loss(_src(pred), _src(nan_target))

    pointwise = loss.calculate_difference(pred, torch.nan_to_num(nan_target))
    expected = pointwise[valid].mean()  # mean over the valid points only
    assert torch.isfinite(out)
    torch.testing.assert_close(out, expected)

    # Not neutral for BCE if the NaN were zero-filled instead of excluded.
    zero_filled = loss(_src(pred), _src(torch.nan_to_num(nan_target)))
    assert not torch.isclose(zero_filled, expected)


@pytest.mark.parametrize("ignore_nans", [True, False])
def test_nan_target_with_nan_prediction_is_excluded(ignore_nans: bool) -> None:
    pred, target = _data()
    nan_target = target.clone()
    nan_target[0, ..., 3, :] = torch.nan
    nan_pred = pred.clone()
    nan_pred[0, ..., 3, :] = torch.nan
    loss = _loss(ignore_nans=ignore_nans)
    torch.testing.assert_close(loss(_src(nan_pred), _src(nan_target)), loss(_src(pred), _src(nan_target)))


def test_nan_prediction_follows_ignore_nans() -> None:
    pred, target = _data()
    nan_pred = pred.clone()
    nan_pred[0, ..., 3, :] = torch.nan

    assert torch.isnan(_loss(ignore_nans=False)(_src(nan_pred), _src(target)))

    keep = torch.ones_like(target, dtype=torch.bool)
    keep[0, ..., 3, :] = False
    loss = _loss(ignore_nans=True)
    expected = loss.calculate_difference(pred, target)[keep].mean()
    torch.testing.assert_close(loss(_src(nan_pred), _src(target)), expected)


@pytest.mark.parametrize("ignore_nans", [True, False])
def test_nan_gradients_are_finite_and_zero_at_excluded_points(ignore_nans: bool) -> None:
    pred, target = _data()
    target[0, ..., 3, :] = torch.nan
    pred = pred.clone().requires_grad_(True)
    out = _loss(focal_gamma=2.0, ignore_nans=ignore_nans)(_src(pred), _src(target))
    out.backward()

    assert torch.isfinite(pred.grad).all()
    assert torch.all(pred.grad[0, ..., 3, :] == 0)
    assert torch.any(pred.grad != 0)


def test_all_nan_target_gives_zero_loss_and_finite_gradient() -> None:
    pred, target = _data()
    pred = pred.clone().requires_grad_(True)
    out = _loss()(_src(pred), _src(torch.full_like(target, torch.nan)))
    out.backward()

    assert out == 0
    assert torch.isfinite(pred.grad).all()


def test_nan_exclusion_per_variable_with_variable_scaler() -> None:
    """A variable with NaN targets is renormalised on its own, the other variable is untouched."""
    pred, target = _data(nvars=2)
    target[0, ..., :6, 1] = torch.nan
    loss = _loss()
    loss.add_scaler(TensorDim.VARIABLE, torch.tensor([1.0, 3.0]), name="variable")

    per_var = loss(_src(pred), _src(target), squash=False)
    pointwise = loss.calculate_difference(pred, torch.nan_to_num(target))
    valid = ~torch.isnan(target)
    torch.testing.assert_close(per_var[0], pointwise[..., 0][valid[..., 0]].mean())
    torch.testing.assert_close(per_var[1], 3.0 * pointwise[..., 1][valid[..., 1]].mean())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_saturated_inputs_give_finite_loss_and_gradients(dtype: torch.dtype) -> None:
    pred = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5], dtype=dtype).reshape(1, 1, 1, 5, 1)
    target = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32).reshape(1, 1, 1, 5, 1)
    pred = pred.clone().requires_grad_(True)
    loss = BinaryCrossEntropyLoss(focal_gamma=2.0, alpha=0.25)
    loss.add_scaler(TensorDim.GRID, torch.ones(5) / 5)

    out = loss(_src(pred), _src(target))
    out.backward()

    assert out.dtype == torch.float32
    assert torch.isfinite(out)
    assert torch.isfinite(pred.grad).all()


def test_cpu_autocast_bfloat16_computes_in_float32() -> None:
    pred, target = _data()
    loss = _loss()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = loss(_src(pred.requires_grad_(True)), _src(target))
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, F.binary_cross_entropy(pred, target), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fp16 autocast")
@pytest.mark.parametrize("focal_gamma", [0.0, 2.0])
@pytest.mark.parametrize("saturated", [False, True])
def test_cuda_fp16_autocast_forward_backward(focal_gamma: float, saturated: bool) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    layer = torch.nn.Linear(4, 1).to(device)
    x = torch.randn(2, 1, 1, GRID, 4, device=device)
    if saturated:
        x = x * 1e4  # the sigmoid saturates to exactly 0 and 1 in fp16
    target = (torch.rand(2, 1, 1, GRID, 1, device=device) > 0.5).float()
    target[0, ..., 0, :] = torch.nan

    loss = BinaryCrossEntropyLoss(focal_gamma=focal_gamma, alpha=0.25, ignore_nans=True).to(device)
    loss.add_scaler(TensorDim.GRID, torch.ones(GRID, device=device) / GRID)

    with torch.autocast("cuda", dtype=torch.float16):
        pred = torch.sigmoid(layer(x))
        assert pred.dtype == torch.float16
        if saturated:
            assert ((pred == 0) | (pred == 1)).any()
        out = loss(_src(pred), _src(target))
    out.backward()

    assert out.dtype == torch.float32
    assert torch.isfinite(out)
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def _data_indices() -> IndexCollection:
    return IndexCollection(
        DictConfig({"forcing": [], "diagnostic": [], "target": []}),
        {"t2m": 0, "tc_mask": 1},
    )


def test_combined_loss_child_on_selected_variable() -> None:
    node_weights = (TensorDim.GRID.value, torch.ones(GRID) / GRID)
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MSELoss",
                        "predicted_variables": ["t2m"],
                        "scalers": ["node_weights"],
                    },
                    {
                        "_target_": TARGET,
                        "predicted_variables": ["tc_mask"],
                        "focal_gamma": 2.0,
                        "alpha": 0.25,
                        "scalers": ["node_weights"],
                        "ignore_nans": True,
                    },
                ],
                "loss_weights": [1.0, 2.0],
            },
        ),
        scalers={"node_weights": node_weights},
        data_indices=_data_indices(),
    )
    assert isinstance(loss, CombinedLoss)

    pred = torch.rand(2, 1, 1, GRID, 2) * 0.98 + 0.01
    target = torch.stack(
        [torch.randn(2, 1, 1, GRID), (torch.rand(2, 1, 1, GRID) > 0.7).float()],
        dim=-1,
    )
    target[0, ..., :3, 1] = torch.nan
    out = loss(
        _src(pred.requires_grad_(True)),
        _src(target),
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.MODEL_OUTPUT,
    )
    out.backward()

    mse = MSELoss()
    mse.add_scaler(TensorDim.GRID, torch.ones(GRID) / GRID)
    bce = _loss(focal_gamma=2.0, alpha=0.25, ignore_nans=True)
    expected = mse(_src(pred[..., :1]), _src(target[..., :1])) + 2.0 * bce(_src(pred[..., 1:]), _src(target[..., 1:]))
    torch.testing.assert_close(out, expected)
    assert torch.isfinite(pred.grad).all()


def test_as_inner_loss_of_time_aggregate_wrapper() -> None:
    inner = BinaryCrossEntropyLoss(ignore_nans=True)
    wrapper = TimeAggregateLossWrapper(time_aggregation_types=["mean"], loss_fn=inner, ignore_nans=True)
    wrapper.add_scaler(TensorDim.GRID, torch.ones(GRID) / GRID)

    pred = torch.rand(2, 3, 1, GRID, 1) * 0.98 + 0.01
    target = (torch.rand(2, 3, GRID, 1) > 0.5).float().unsqueeze(2)
    target[0, ..., 2, :] = torch.nan
    out = wrapper(_src(pred), _src(target))

    expected = inner(_src(pred.mean(dim=1, keepdim=True)), _src(target.mean(dim=1, keepdim=True)))
    assert torch.isfinite(out)
    torch.testing.assert_close(out, expected)


def test_tabular_nan_targets_are_averaged_over_valid_observations() -> None:
    pred = torch.tensor([[0.9, 0.2], [0.4, 0.6], [0.3, 0.7]])
    target = torch.tensor([[1.0, 0.0], [torch.nan, 1.0], [0.0, torch.nan]])

    def view(data: torch.Tensor) -> TabularSource:
        return TabularSource(
            name="obs",
            values=[data],
            variables=["a", "b"],
            statistics={},
            coordinates=[torch.zeros(data.shape[0], 2)],
            layout=TensorLayout(grid=0, variables=1),
            boundaries=[(slice(0, data.shape[0]),)],
            timedeltas=[torch.zeros(data.shape[0])],
        )

    loss = BinaryCrossEntropyLoss()
    out = loss(view(pred), view(target), squash=False)

    pointwise = loss.calculate_difference(pred, torch.nan_to_num(target))
    valid = ~torch.isnan(target)
    expected = torch.stack([pointwise[valid[:, i], i].mean() for i in range(2)])
    torch.testing.assert_close(out, expected)


def test_weights_multiply_pointwise_loss() -> None:
    pred, target = _data(nvars=2)
    weights = torch.tensor([1.0, 3.0]).view(1, 1, 1, 1, 2)
    loss = _loss()
    weighted = loss(_src(pred), _src(target), weights=weights, squash=False)
    plain = loss(_src(pred), _src(target), squash=False)
    torch.testing.assert_close(weighted, plain * torch.tensor([1.0, 3.0]))


def test_loss_schema_round_trip() -> None:
    config = {
        "_target_": TARGET,
        "scalers": ["node_weights"],
        "ignore_nans": True,
        "predicted_variables": ["tc_mask"],
        "focal_gamma": 2.0,
        "alpha": 0.25,
        "pos_weight": 3.0,
        "eps": 1e-5,
    }
    adapter = TypeAdapter(LossSchemas)
    schema = adapter.validate_python(config)
    assert isinstance(schema, BinaryCrossEntropyLossSchema)
    assert (schema.focal_gamma, schema.alpha, schema.pos_weight, schema.eps) == (2.0, 0.25, 3.0, 1e-5)

    dumped = adapter.dump_python(schema, by_alias=True, mode="json")
    assert dumped["_target_"] == TARGET
    assert adapter.validate_python(dumped) == schema

    # The validated parameters are accepted by the loss itself.
    params = {k: v for k, v in dumped.items() if k in {"focal_gamma", "alpha", "pos_weight", "eps"}}
    loss = BinaryCrossEntropyLoss(ignore_nans=True, **params)
    assert (loss.focal_gamma, loss.alpha, loss.pos_weight, loss.eps) == (2.0, 0.25, 3.0, 1e-5)


def test_loss_schema_defaults_and_combined_child() -> None:
    schema = TypeAdapter(LossSchemas).validate_python({"_target_": TARGET, "scalers": []})
    assert (schema.focal_gamma, schema.alpha, schema.pos_weight) == (0.0, None, 1.0)

    combined = CombinedLossSchema.model_validate(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [
                {"_target_": "anemoi.training.losses.MSELoss", "scalers": []},
                {"_target_": TARGET, "scalers": [], "focal_gamma": 1.5, "predicted_variables": ["tc_mask"]},
            ],
        },
    )
    assert isinstance(combined.losses[1], BinaryCrossEntropyLossSchema)
    assert combined.losses[1].focal_gamma == 1.5


@pytest.mark.parametrize(
    "kwargs",
    [{"focal_gamma": -0.1}, {"alpha": 1.1}, {"alpha": -0.1}, {"pos_weight": 0.0}, {"eps": 0.5}, {"eps": 0.0}],
)
def test_loss_schema_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python({"_target_": TARGET, "scalers": [], **kwargs})
