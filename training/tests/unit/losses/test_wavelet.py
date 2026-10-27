# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from pydantic import ValidationError
from torch.nn import functional

from anemoi.models.layers.wavelet_transforms import WaveletPyramid
from anemoi.training.losses import RegionalWaveletLoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.schemas.training import RegionalWaveletLossSchema
from anemoi.training.utils.enums import TensorDim

pytest.importorskip("pytorch_wavelets")


@pytest.fixture(autouse=True, scope="module")
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def isolated_compile_cache() -> Iterator[None]:
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _data(
    shape: tuple[int, int] = (8, 10),
    batch: int = 1,
    time: int = 1,
    ensemble: int = 1,
    variables: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    dims = (batch, time, ensemble, shape[0] * shape[1], variables)
    return torch.randn(dims, generator=generator), torch.randn(dims, generator=generator)


def _manual_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: tuple[int, int],
    levels: int,
    scale_weights: list[float],
    ignore_nans: bool = False,
    spatial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent reference using native coefficients and complex inner products."""
    from pytorch_wavelets import DTCWTForward

    b, t, e, _, v = pred.shape
    p = pred.permute(0, 1, 2, 4, 3).reshape(-1, 1, *shape)
    y = target.expand_as(pred).permute(0, 1, 2, 4, 3).reshape_as(p)
    if spatial is None:
        w = torch.full_like(p, 1 / (shape[0] * shape[1]))
    else:
        w = spatial.expand_as(pred).permute(0, 1, 2, 4, 3).reshape_as(p)
    included = w > 0
    valid = (~included | (torch.isfinite(p) & torch.isfinite(y))).float() if ignore_nans else None
    if valid is not None:
        included = included & valid.bool()
    p, y = torch.where(included, p, 0), torch.where(included, y, 0)
    total_spatial_weight = w.flatten(1).sum(1)
    relative_spatial_weights = w / (total_spatial_weight[:, None, None, None] / (shape[0] * shape[1])).clamp_min(1e-30)
    transform = DTCWTForward(J=levels, biort="near_sym_b", qshift="qshift_b")
    lp, hp = transform(p)
    lt, ht = transform(y.detach())
    total = torch.zeros_like(total_spatial_weight)
    mask = valid
    for weight, cp, ct in zip(scale_weights, hp, ht, strict=True):
        cp, ct = torch.view_as_complex(cp.contiguous()), torch.view_as_complex(ct.contiguous())
        resolution = cp.shape[-2:]
        pooled = functional.adaptive_avg_pool2d(relative_spatial_weights, resolution)
        pooled = pooled / pooled.mean((-2, -1), keepdim=True).clamp_min(1e-30)
        if mask is not None:
            mask = functional.adaptive_avg_pool2d(mask, resolution)
            pooled = pooled * mask
        pooled = pooled.unsqueeze(2)
        pp, pt = cp.abs().square(), ct.abs().square()
        cross = (cp * ct.conj()).real
        pp, pt, cross = [(a * pooled).sum((-2, -1)) for a in (pp, pt, cross)]
        a, b_amp = (pp + 1e-8).sqrt(), (pt + 1e-8).sqrt()
        error = (a - b_amp).square() + 2 * torch.maximum(pp, pt) * (1 - cross / (a * b_amp + 1e-8))
        total = total + weight * error.flatten(1).sum(1)
    pooled = functional.adaptive_avg_pool2d(relative_spatial_weights, lp.shape[-2:])
    pooled = pooled / pooled.mean((-2, -1), keepdim=True).clamp_min(1e-30)
    if mask is not None:
        pooled = pooled * functional.adaptive_avg_pool2d(mask, lp.shape[-2:])
    total = total + ((lp - lt).square() * pooled).flatten(1).sum(1)
    denom = shape[0] * shape[1] if valid is None else valid.flatten(1).sum(1).clamp_min(1e-6)
    return (total * total_spatial_weight / denom).reshape(b, t, e, v).sum(1).mean((0, 1))


@pytest.mark.parametrize("shape", [(8, 10), (7, 9)])
@pytest.mark.parametrize("ignore_nans", [False, True])
def test_numerical_values_and_gradients_match_native_reference(
    shape: tuple[int, int],
    ignore_nans: bool,
) -> None:
    pred, target = _data(shape, batch=2, time=2, ensemble=2)
    if ignore_nans:
        target[:, :, :, ::9] = float("nan")
    pred.requires_grad_()
    loss = RegionalWaveletLoss(shape, J=2, scale_weights=[0.25, 1.5], ignore_nans=ignore_nans)
    actual = loss(pred, target, squash=False)
    expected = _manual_reference(pred, target, shape, 2, [0.25, 1.5], ignore_nans)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
        rtol=2e-4,
        atol=2e-6,
    )


def test_spatial_weighted_numerics() -> None:
    shape = (7, 9)
    pred, target = _data(shape)
    pred.requires_grad_()
    weights = torch.linspace(0.1, 2, 63)
    weights[::5] = 0
    weights = 0.33 * weights / weights.sum()
    loss = RegionalWaveletLoss(shape, J=2)
    loss.add_scaler(TensorDim.GRID, weights, name="area")
    actual = loss(pred, target, squash=False)
    expected = _manual_reference(pred, target, shape, 2, [1, 1], spatial=weights[None, None, None, :, None])
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
        rtol=2e-4,
        atol=2e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ignore_nans", [False, True])
def test_chunked_mixed_spatial_scalers_match_dense_reference(dtype: torch.dtype, ignore_nans: bool) -> None:
    pred, target = _data((8, 9), batch=2, time=2, ensemble=2, variables=3)
    pred, target = pred.to(dtype), target[:, :, :1].to(dtype)
    if ignore_nans:
        pred[:, :, :, 3::9] = float("inf")
        target[:, :, :, 1::8] = float("nan")
    pred = pred.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    target.requires_grad_()
    area = torch.linspace(0.1, 1, 72) / 72
    time_grid = torch.linspace(0.2, 1, 144).reshape(2, 72)
    mask = torch.ones(2, 2, 72, 3)
    mask[0, :, ::3, 0] = 0
    mask[1, 1, 1::4, 2] = 0
    originals = [value.clone() for value in (area, time_grid, mask)]
    loss = RegionalWaveletLoss([7, 9], J=2, subgrid=(4, 67), ignore_nans=ignore_nans, xfm_batch_chunk_size=5)
    loss.add_scaler(3, area, name="area")
    loss.add_scaler((1, 3), time_grid, name="time_grid")
    loss.add_scaler((0, 2, 3, 4), mask, name="mask")
    spatial = area[None, None, None, :, None] * time_grid[None, :, None, :, None] * mask[:, None]
    actual = loss(pred, target, squash=False)
    expected = _manual_reference(
        pred[:, :, :, 4:67].float(),
        target[:, :, :, 4:67].float(),
        (7, 9),
        2,
        [1, 1],
        ignore_nans,
        spatial=spatial[:, :, :, 4:67],
    )
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
        rtol=0.01 if dtype == torch.bfloat16 else 2e-4,
        atol=2e-5 if dtype == torch.bfloat16 else 2e-6,
    )
    assert torch.autograd.grad(actual.sum(), target, allow_unused=True)[0] is None
    for value, original in zip((area, time_grid, mask), originals, strict=True):
        torch.testing.assert_close(value, original)


def test_spatial_scalers_stay_compact_and_uniform_weights_are_implicit() -> None:
    pred, _ = _data(batch=2, time=3, ensemble=4, variables=3)
    loss = RegionalWaveletLoss([8, 10], J=2)
    weights, non_spatial_weights = loss._weights(pred, None, None, None, None)
    assert not weights
    assert non_spatial_weights.numel() == 1
    indices = torch.unravel_index(torch.arange(5), (2, 3, 4, 3))
    relative_spatial_weights, total_spatial_weight, included = loss._chunk_weights(weights, indices)
    assert relative_spatial_weights.shape == (5, 1, 1, 1)
    assert included is None
    assert loss._pool_spatial_weights(relative_spatial_weights, (4, 5)) is relative_spatial_weights
    torch.testing.assert_close(total_spatial_weight, torch.ones(5))
    area = torch.ones(80) / 80
    mask = torch.ones(2, 80, 3)
    loss.add_scaler(3, area, name="area")
    loss.add_scaler((0, 3, 4), mask, name="mask")
    weights, _ = loss._weights(pred, None, None, None, None)
    assert [value.shape for value in weights] == [(1, 1, 1, 80, 1), (2, 1, 1, 80, 3)]
    assert sum(value.numel() for value in weights) == area.numel() + mask.numel()


def test_field_gathering_and_fp32_casting_are_chunk_local(monkeypatch: pytest.MonkeyPatch) -> None:
    pred, target = _data((10, 9), batch=2, time=2, ensemble=2, variables=3)
    pred = pred.bfloat16().transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    target = target[:, :, :1].bfloat16()
    loss = RegionalWaveletLoss([7, 9], J=2, subgrid=(5, 68), xfm_batch_chunk_size=5)
    gather = loss._chunk_fields
    chunk_sizes = []

    def record(values: torch.Tensor, indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        assert values.dtype == torch.bfloat16
        assert values.shape[3] == 63
        result = gather(values, indices)
        assert result.dtype == torch.float32
        assert result.shape[1:] == (1, 7, 9)
        chunk_sizes.append(result.shape[0])
        return result

    monkeypatch.setattr(loss, "_chunk_fields", record)
    actual = loss(pred, target)
    expected = RegionalWaveletLoss([7, 9], J=2)(pred[:, :, :, 5:68].float(), target[:, :, :, 5:68].float())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])
    assert chunk_sizes == [5, 5, 5, 5, 5, 5, 5, 5, 4, 4]


def test_chunked_spatial_scaler_product_overflow_is_rejected() -> None:
    pred, target = _data()
    loss = RegionalWaveletLoss([8, 10], J=2, xfm_batch_chunk_size=1)
    loss.add_scaler(3, torch.full((80,), 1e30), name="large")
    loss.add_scaler((3, 4), torch.full((80, 2), 1e20), name="also_large")
    with pytest.raises(ValueError, match=r"products.*finite"):
        loss(pred, target)


def test_power_and_cross_terms_are_summed_before_amse() -> None:
    loss = RegionalWaveletLoss([2, 2], J=1)
    pred = torch.zeros(1, 1, 6, 1, 2, 2)
    target = torch.zeros_like(pred)
    pred[0, 0, 0, 0, 0, 0] = 1
    target[0, 0, 0, 0, 1, 0] = 1
    lowpass = torch.zeros(1, 1, 2, 2)
    actual = loss._score_pyramid(
        WaveletPyramid(lowpass, (pred,)),
        WaveletPyramid(lowpass, (target,)),
        torch.ones_like(lowpass),
        None,
    )
    torch.testing.assert_close(actual, torch.tensor([0.5]))


def test_unit_sum_weights_and_absolute_mass() -> None:
    pred, target = _data((7, 9))
    pred, target = pred * 1e-4, target * 1e-4
    loss = RegionalWaveletLoss([7, 9], J=2)
    original = loss(pred, target)
    loss.add_scaler(3, torch.ones(63) / 63, name="area")
    torch.testing.assert_close(loss(pred, target), original)
    loss.update_scaler("area", torch.ones(63) * (0.33 / 63))
    torch.testing.assert_close(loss(pred, target), original * 0.33, rtol=3e-6, atol=1e-14)


def test_spatial_weight_extremes() -> None:
    pred, target = _data()
    loss = RegionalWaveletLoss([8, 10], J=2)
    original = loss(pred, target)
    weights = torch.full((80,), 1e-40)
    loss.add_scaler(3, weights, name="area")
    torch.testing.assert_close(loss(pred, target) / weights.sum(), original, rtol=2e-5, atol=1e-6)
    loss.update_scaler("area", torch.full((80,), 1e38))
    with pytest.raises(ValueError, match=r"total.*finite"):
        loss(pred, target)


def test_nonuniform_weights_and_runtime_masks() -> None:
    pred, target = _data(batch=2, variables=2)
    loss = RegionalWaveletLoss([8, 10], J=2)
    loss.add_scaler(3, torch.ones(80) / 80, name="area")
    uniform = loss(pred, target)
    area = torch.linspace(0.1, 3, 80)
    loss.update_scaler("area", area / area.sum())
    assert not torch.isclose(loss(pred, target), uniform)
    mask = torch.ones(2, 80, 2)
    mask[:, ::4, 0] = 0
    loss.add_scaler((0, 3, 4), mask, name="runtime_mask")
    baseline = loss(pred, target)
    changed = pred.clone()
    changed[:, :, :, ::4, 0] = 99999
    torch.testing.assert_close(loss(changed, target), baseline)
    loss.update_scaler("runtime_mask", torch.ones_like(mask))
    assert not torch.isclose(loss(changed, target), baseline)
    loss.update_scaler("runtime_mask", torch.zeros_like(mask))
    p = pred.clone().requires_grad_()
    empty = loss(p, target)
    assert empty.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(empty, p)[0]) == 0


def test_invalid_pairs_and_all_invalid_gradients() -> None:
    pred, target = _data()
    target[:, :, :, ::3] = float("nan")
    pred[:, :, :, 1::5] = float("inf")
    loss = RegionalWaveletLoss([8, 10], J=2, ignore_nans=True)
    p = pred.clone().requires_grad_()
    out = loss(p, target)
    assert torch.isfinite(out)
    assert torch.isfinite(torch.autograd.grad(out, p)[0]).all()
    changed = pred.clone()
    changed[torch.isnan(target)] = -12345
    torch.testing.assert_close(loss(changed, target), out)
    all_invalid = torch.full_like(target, float("nan"))
    out = loss(p, all_invalid)
    assert out.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(out, p)[0]) == 0
    assert torch.isnan(RegionalWaveletLoss([8, 10], J=2)(pred, target))


def test_lowpass_only_constant_field_and_finite_difference_gradient() -> None:
    loss = RegionalWaveletLoss([8, 10], J=1, scale_weights=[0])
    pred = torch.ones(1, 1, 1, 80, 1, requires_grad=True)
    target = torch.zeros_like(pred)
    torch.testing.assert_close(loss(pred, target), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
    pred, target = _data(variables=1)
    pred.requires_grad_()
    loss = RegionalWaveletLoss([8, 10], J=2)
    direction = torch.randn_like(pred)
    derivative = (torch.autograd.grad(loss(pred, target), pred)[0] * direction).sum()
    step = 1e-3
    finite_difference = (loss(pred + step * direction, target) - loss(pred - step * direction, target)) / (2 * step)
    torch.testing.assert_close(derivative, finite_difference, rtol=0.01, atol=3e-4)


@pytest.mark.parametrize(("separate", "chunk"), [(False, None), (True, None), (False, 3), (True, 1)])
def test_chunking_broadcasting_and_target_detachment(separate: bool, chunk: int | None) -> None:
    pred, target = _data(batch=2, time=2, ensemble=3)
    target = target[:, :, 0].clone().requires_grad_()
    pred.requires_grad_()
    baseline = RegionalWaveletLoss([8, 10], J=2)(pred, target)
    actual = RegionalWaveletLoss(
        [8, 10],
        J=2,
        separate_pred_target_xfm=separate,
        xfm_batch_chunk_size=chunk,
    )(pred, target)
    torch.testing.assert_close(actual, baseline)
    torch.testing.assert_close(
        torch.autograd.grad(actual, pred)[0],
        torch.autograd.grad(baseline, pred)[0],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.autograd.grad(actual, target, allow_unused=True)[0] is None


def test_subgrid_scalers_and_reductions() -> None:
    pred, target = _data((9, 10), time=2, ensemble=2, variables=3)
    loss = RegionalWaveletLoss([8, 10], J=2, subgrid=(5, 85))
    reference = RegionalWaveletLoss([8, 10], J=2)
    loss.add_scaler(3, torch.ones(90) / 90, name="area")
    reference.add_scaler(3, torch.ones(80) / 90, name="area")
    loss.add_scaler(4, torch.tensor([1.0, 2.0, 3.0]), name="variable")
    reference.add_scaler(4, torch.tensor([1.0, 2.0, 3.0]), name="variable")
    loss.add_scaler(1, torch.tensor([0.25, 0.75]), name="time")
    reference.add_scaler(1, torch.tensor([0.25, 0.75]), name="time")
    expected = reference(pred[:, :, :, 5:85], target[:, :, :, 5:85], squash=False)
    actual = loss(pred, target, squash=False)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(loss(pred, target), actual.mean())
    torch.testing.assert_close(loss(pred, target, squash_mode="sum"), actual.sum())
    torch.testing.assert_close(loss(pred, target, squash=False, scaler_indices=(..., [2, 0])), actual[[2, 0]])
    torch.testing.assert_close(
        loss(pred, target, without_scalers=["variable"], squash=False),
        actual / torch.tensor([1, 2, 3]),
    )
    torch.testing.assert_close(
        loss(pred, target, without_scalers=[4], squash=False),
        actual / torch.tensor([1, 2, 3]),
    )
    torch.testing.assert_close(
        loss(pred, target, without_scalers=["area"]),
        loss(pred, target, without_scalers=[3]),
    )


def test_autocast_and_no_grad_contexts() -> None:
    pred, target = _data()
    pred.requires_grad_()
    loss = RegionalWaveletLoss([8, 10], J=2)
    expected = loss(pred, target)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss(pred, target)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        assert not loss(pred, target).requires_grad


@pytest.mark.parametrize("value", [0.0, 1.0, 1e-5])
def test_identical_and_zero_fields_have_stable_gradients(value: float) -> None:
    pred = torch.full((1, 1, 1, 80, 1), value, requires_grad=True)
    target = pred.detach().clone()
    loss = RegionalWaveletLoss([8, 10], J=2)
    out = loss(pred, target)
    assert torch.isfinite(out) and out >= -1e-6
    assert torch.isfinite(torch.autograd.grad(out, pred)[0]).all()
    assert out < 1e-5


def test_default_levels_and_transform_configuration() -> None:
    pred, target = _data((32, 48))
    automatic = RegionalWaveletLoss([32, 48], transform={"biort": "near_sym_a", "qshift": "qshift_a"})
    explicit = RegionalWaveletLoss([32, 48], J=2, transform={"biort": "near_sym_a", "qshift": "qshift_a"})
    assert automatic.J == 2
    torch.testing.assert_close(automatic(pred, target), explicit(pred, target))
    assert RegionalWaveletLoss([32, 48], separate_pred_target_xfm_threshold=100).separate_pred_target_xfm


def test_scaler_dimension_order_broadcasting_and_exclusions() -> None:
    pred, target = _data(variables=2)
    mask = torch.ones(80, 2)
    mask[::3, 0] = 0
    reference = RegionalWaveletLoss([8, 10], J=2)
    reordered = RegionalWaveletLoss([8, 10], J=2)
    for loss in (reference, reordered):
        loss.add_scaler(3, torch.ones(80) / 80, name="area")
    reference.add_scaler((3, 4), mask, name="mask")
    reordered.add_scaler((-1, -2), mask.T, name="mask")
    torch.testing.assert_close(reordered(pred, target), reference(pred, target))
    torch.testing.assert_close(
        reordered(pred, target, without_scalers=[4]),
        reference(pred, target, without_scalers=["mask"]),
    )
    with pytest.raises(ValueError, match="five-dimensional"):
        reordered.add_scaler(5, torch.ones(80), name="invalid_axis")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reordered.add_scaler(3, -torch.ones(80, 2), name="invalid_contiguous_axes")
    reordered.add_scaler(0, torch.ones(3), name="invalid_batch")
    with pytest.raises(ValueError, match="does not broadcast"):
        reordered(pred, target)


def test_spatial_weights_and_finite_pair_mask_are_both_applied() -> None:
    pred, target = _data()
    target[:, :, :, 3::4] = float("nan")
    weights = torch.linspace(0.1, 1, 80)
    weights[::4] = 0
    loss = RegionalWaveletLoss([8, 10], J=2, ignore_nans=True)
    loss.add_scaler(3, weights / weights.sum(), name="area")
    expected = loss(pred, target)
    changed = pred.clone()
    changed[:, :, :, ::4] = float("nan")
    changed[:, :, :, 3::4] = 99999
    torch.testing.assert_close(loss(changed, target), expected)


def test_loss_variable_mapper_aligns_spatial_masks_and_variable_weights() -> None:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.utils.index_space import IndexSpace

    indices = IndexCollection(OmegaConf.create({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1, "c": 2})
    mask = torch.ones(1, 80, 3)
    mask[:, ::4, 2] = 0
    scalers = {
        "area": (3, torch.ones(80) / 80),
        "mask": ((0, 3, 4), mask),
        "variable": (4, torch.tensor([1.0, 2.0, 3.0])),
    }
    mapped = get_loss_function(
        OmegaConf.create(
            {
                "_target_": "anemoi.training.losses.RegionalWaveletLoss",
                "grid_shape": [8, 10],
                "J": 2,
                "predicted_variables": ["c", "a"],
                "scalers": list(scalers),
            },
        ),
        scalers,
        data_indices=indices,
    )
    pred, target = _data(variables=3)
    direct = RegionalWaveletLoss([8, 10], J=2)
    direct.add_scaler(3, scalers["area"][1], name="area")
    direct.add_scaler((0, 3, 4), mask[..., [2, 0]], name="mask")
    direct.add_scaler(4, torch.tensor([3.0, 1.0]), name="variable")
    torch.testing.assert_close(
        mapped(pred, target, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.MODEL_OUTPUT),
        direct(pred[..., [2, 0]], target[..., [2, 0]]),
    )


def test_anemoi_compile_configuration_selects_the_scoring_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from anemoi.models.utils.compile import mark_for_compilation

    calls = []

    def compile_kernel(kernel: Callable[..., torch.Tensor], **options: Any) -> Callable[..., torch.Tensor]:
        calls.append((kernel, options))
        return kernel

    monkeypatch.setattr("anemoi.models.utils.compile._meets_library_versions_for_compile", lambda: True)
    monkeypatch.setattr(torch, "compile", compile_kernel)
    model = torch.nn.Module()
    loss = RegionalWaveletLoss([8, 10], J=2)
    model.add_module("loss", loss)
    config = OmegaConf.create(
        [{"module": "anemoi.training.losses.RegionalWaveletLoss", "options": {"fullgraph": True, "dynamic": False}}],
    )
    mark_for_compilation(model, config)
    assert calls == [(loss._score_pyramid, {"fullgraph": True, "dynamic": False})]


@pytest.mark.parametrize(
    "options",
    [
        {"grid_shape": [1, 8]},
        {"J": 0},
        {"eps": 0},
        {"scale_weights": [1]},
        {"scale_weights": [1, -1]},
        {"subgrid": "output_mask"},
        {"subgrid": (5, 2)},
        {"xfm_batch_chunk_size": 0},
    ],
)
def test_invalid_configuration(options: dict[str, Any]) -> None:
    kwargs = {"grid_shape": [8, 10], "J": 2, **options}
    with pytest.raises((ValueError, TypeError)):
        RegionalWaveletLoss(**kwargs)


@pytest.mark.parametrize("removed_mode", ["band_summed", "per_point"])
def test_mode_selector_is_not_supported(removed_mode: str) -> None:
    options: dict[str, Any] = {"grid_shape": [8, 10], "J": 2, "psd_mode": removed_mode}
    with pytest.raises(TypeError, match="psd_mode"):
        RegionalWaveletLoss(**options)
    with pytest.raises(ValidationError, match="psd_mode"):
        TypeAdapter(LossSchemas).validate_python({"_target_": "anemoi.training.losses.RegionalWaveletLoss", **options})


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_spatial_scalers_rejected_on_add_and_update(value: float) -> None:
    loss = RegionalWaveletLoss([8, 10], J=2)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        loss.add_scaler(3, torch.full((80,), value), name="bad")
    loss.add_scaler(3, torch.ones(80), name="area")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        loss.update_scaler("area", torch.full((80,), value))


def test_input_and_shard_validation() -> None:
    pred, target = _data()
    loss = RegionalWaveletLoss([8, 10], J=2)
    for kwargs in (
        {"grid_shard_slice": slice(0, 80)},
        {"grid_shard_sizes": [80]},
        {"grid_shard_slice": slice(0, 80), "grid_shard_sizes": [80]},
        {"grid_dim": 0},
        {"scaler_indices": [0]},
    ):
        with pytest.raises((ValueError, TypeError)):
            loss(pred, target, **kwargs)
    with pytest.raises(ValueError, match="point count"):
        RegionalWaveletLoss([8, 10], subgrid=(1, 82))(pred, target)
    with pytest.raises(ValueError, match="point count"):
        RegionalWaveletLoss([8, 10])(pred[:, :, :, :-1], target[:, :, :, :-1])
    with pytest.raises(ValueError, match="shapes must match"):
        loss(pred, target[..., :1])
    with pytest.raises(TypeError, match="floating-point"):
        loss(pred.long(), target.long())
    loss.add_scaler(3, torch.ones(79), name="bad_shape")
    with pytest.raises(ValueError, match="input grid"):
        loss(pred, target)


def test_schema_factory_combination_and_output_mask_resolution() -> None:
    from anemoi.training.train.methods.base import resolve_subgrid

    config = {
        "_target_": "anemoi.training.losses.RegionalWaveletLoss",
        "grid_shape": [8, 10],
        "J": 2,
        "subgrid": "output_mask",
        "scalers": ["area"],
        "transform": {"backend": "pytorch_wavelets"},
    }
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert isinstance(validated, RegionalWaveletLossSchema)
    data_config = OmegaConf.create({"regional": config})

    class Mask:
        def as_tuple(self) -> tuple[int, int]:
            return (5, 85)

    resolve_subgrid(data_config, {"regional": Mask()})
    standalone = get_loss_function(data_config.regional, {"area": (3, torch.ones(90) / 90)})
    combined_config = OmegaConf.create(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [OmegaConf.to_container(data_config.regional)],
            "loss_weights": [0.4],
        },
    )
    TypeAdapter(LossSchemas).validate_python(OmegaConf.to_container(combined_config))
    combined = get_loss_function(combined_config, {"area": (3, torch.ones(90) / 90)})
    pred, target = _data((9, 10))
    torch.testing.assert_close(combined(pred, target), 0.4 * standalone(pred, target))


@pytest.mark.usefixtures("isolated_compile_cache")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_real_reference_compiled_score_forward_backward_and_mask_update(backend: str) -> None:
    pred, target = _data((7, 9), ensemble=2)
    target[:, :, :, ::8] = float("nan")
    options = {"grid_shape": [7, 9], "J": 2, "ignore_nans": True, "xfm_batch_chunk_size": 2}
    eager, compiled = RegionalWaveletLoss(**options), RegionalWaveletLoss(**options)
    for loss in (eager, compiled):
        loss.add_scaler(3, torch.linspace(0.1, 2, 63) / 63, name="area")
        loss.add_scaler((0, 3, 4), torch.ones(1, 63, 2), name="mask")
    compiled.compile_for_training(backend=backend, fullgraph=True, dynamic=False)
    for i in range(2):
        mask = torch.ones(1, 63, 2)
        mask[:, i::4] = 0
        for loss in (eager, compiled):
            loss.update_scaler("mask", mask)
        p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
        a, b = eager(p1, target), compiled(p2, target)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, p1)[0], torch.autograd.grad(b, p2)[0])


@pytest.mark.usefixtures("isolated_compile_cache")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("ignore_nans", [False, True])
def test_compiled_uniform_spatial_weights_and_partial_chunks(backend: str, ignore_nans: bool) -> None:
    pred, target = _data((7, 9), ensemble=2)
    if ignore_nans:
        target[:, :, :, ::8] = float("nan")
    options = {"grid_shape": [7, 9], "J": 2, "ignore_nans": ignore_nans, "xfm_batch_chunk_size": 3}
    eager, compiled = RegionalWaveletLoss(**options), RegionalWaveletLoss(**options)
    compiled.compile_for_training(backend=backend, fullgraph=True, dynamic=False)
    p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    actual, expected = compiled(p1, target), eager(p2, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, p1)[0], torch.autograd.grad(expected, p2)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton DTCWT requires a GPU")
@pytest.mark.usefixtures("isolated_compile_cache")
def test_real_triton_eager_compiled_loss() -> None:
    pytest.importorskip("triton_dtcwt")
    pred, target = (t.cuda() for t in _data((16, 20), variables=1))
    options = {"grid_shape": [16, 20], "J": 2, "transform": {"backend": "triton"}}
    eager, compiled = RegionalWaveletLoss(**options), RegionalWaveletLoss(**options)
    compiled.compile_for_training(fullgraph=True, dynamic=False)
    p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    actual, expected = compiled(p1, target), eager(p2, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, p1)[0], torch.autograd.grad(expected, p2)[0])


def _distributed_worker(rank: int, rendezvous: str, variables: int, local_mask: bool) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=60),
    )
    try:
        pred, target = _data((6, 9), variables=variables, ensemble=2)
        options = {"grid_shape": [6, 8], "J": 2, "subgrid": (3, 51), "xfm_batch_chunk_size": 1}
        full_loss = RegionalWaveletLoss(**options)
        loss = RegionalWaveletLoss(**options)
        mask = torch.ones(1, 54, variables)
        mask[:, ::4] = 0
        sizes = [19, 35]
        start = sum(sizes[:rank])
        shard_slice = slice(start, start + sizes[rank])
        for item in (full_loss, loss):
            item.add_scaler(3, torch.ones(54) / 54, name="area")
            item.add_scaler(4, torch.arange(1, variables + 1).float(), name="variable")
        full_loss.add_scaler((0, 3, 4), mask, name="mask")
        loss.add_scaler((0, 3, 4), mask[:, shard_slice] if local_mask else mask, name="mask")
        full_pred = pred.clone().requires_grad_()
        expected = full_loss(full_pred, target)
        expected.backward()
        local_pred = pred[:, :, :, shard_slice].clone().requires_grad_()
        actual = loss(
            local_pred,
            target[:, :, :, shard_slice],
            grid_shard_slice=shard_slice,
            grid_shard_sizes=sizes,
            group=torch.distributed.group.WORLD,
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(local_pred.grad, full_pred.grad[:, :, :, shard_slice], atol=2e-6, rtol=2e-5)
        loss.update_scaler("mask", mask)
        torch.testing.assert_close(loss(pred, target, group=torch.distributed.group.WORLD), expected)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize(
    ("variables", "local_mask"),
    [(3, False), (3, True), (1, True)],
)
def test_multiprocess_uneven_grid_and_variable_shards(
    tmp_path: Path,
    variables: int,
    local_mask: bool,
) -> None:
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), variables, local_mask),
        nprocs=2,
        join=True,
    )
