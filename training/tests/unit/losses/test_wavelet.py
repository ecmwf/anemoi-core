# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
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
from torch_geometric.data import HeteroData

from anemoi.models.layers.wavelet_transforms import WaveletPyramid
from anemoi.training.losses import WaveletAMSELoss
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.schemas.training import ImplementedLossesUsingBaseLossSchema
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.schemas.training import SpectralProjectionConfigSchema
from anemoi.training.schemas.training import WaveletAMSELossSchema

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
) -> torch.Tensor:
    """Independent reference using native coefficients and complex inner products."""
    from pytorch_wavelets import DTCWTForward

    b, t, e, _, v = pred.shape
    p = pred.permute(0, 1, 2, 4, 3).reshape(-1, 1, *shape)
    y = target.permute(0, 1, 2, 4, 3).reshape_as(p)
    transform = DTCWTForward(J=levels, biort="near_sym_b", qshift="qshift_b")
    lp, hp = transform(p)
    lt, ht = transform(y.detach())
    total = p.new_zeros(p.shape[0])
    for scale_weight, cp, ct in zip(scale_weights, hp, ht, strict=True):
        cp, ct = torch.view_as_complex(cp.contiguous()), torch.view_as_complex(ct.contiguous())
        pp, pt = cp.abs().square(), ct.abs().square()
        cross = (cp * ct.conj()).real
        pp, pt, cross = [a.sum((-2, -1)) for a in (pp, pt, cross)]
        a, b_amp = (pp + 1e-8).sqrt(), (pt + 1e-8).sqrt()
        error = (a - b_amp).square() + 2 * torch.maximum(pp, pt) * (1 - cross / (a * b_amp + 1e-8))
        total = total + scale_weight * error.flatten(1).sum(1)
    lowpass_error = (lp - lt).square()
    total = total + lowpass_error.flatten(1).sum(1)
    return (total / (shape[0] * shape[1])).reshape(b, t, e, v).sum(1).mean((0, 1))


def test_public_wavelet_amse_name_and_factory_target() -> None:
    from anemoi.training import losses

    target = "anemoi.training.losses.WaveletAMSELoss"
    config = {"_target_": target, "grid_shape": [8, 10], "J": 2}
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert isinstance(validated, WaveletAMSELossSchema)
    assert validated.target_ == target
    assert ImplementedLossesUsingBaseLossSchema.wavelet_amse.value == target
    assert losses.WaveletAMSELoss is WaveletAMSELoss
    assert "WaveletAMSELoss" in losses.__all__
    loss = get_loss_function(OmegaConf.create(config))
    assert isinstance(loss, WaveletAMSELoss)
    assert loss.name == "wavelet_amse"


@pytest.mark.parametrize("shape", [(8, 10), (7, 9)])
def test_numerical_values_and_gradients_match_native_reference(shape: tuple[int, int]) -> None:
    pred, target = _data(shape, batch=2, time=2, ensemble=2)
    pred.requires_grad_()
    loss = WaveletAMSELoss(shape, J=2, scale_weights=[0.25, 1.5])
    actual = loss(pred, target, squash=False)
    expected = _manual_reference(pred, target, shape, 2, [0.25, 1.5])
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
        rtol=2e-4,
        atol=2e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_nonspatial_scaler_matches_dense_reference(dtype: torch.dtype) -> None:
    pred, target = _data((8, 9), batch=2, time=2, ensemble=2, variables=3)
    pred, target = pred.to(dtype), target.to(dtype)
    pred = pred.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    target.requires_grad_()
    variable_weights = torch.tensor([1.0, 2.0, 3.0])
    loss = WaveletAMSELoss([7, 9], J=2, subgrid=(4, 67), transform_field_chunk_size=5)
    loss.add_scaler(4, variable_weights, name="variable")
    actual = loss(pred, target, squash=False)
    expected = torch.cat(
        [
            _manual_reference(
                pred[:, :, :, 4:67, variable : variable + 1].float(),
                target[:, :, :, 4:67, variable : variable + 1].float(),
                (7, 9),
                2,
                [1, 1],
            )
            * variable_weights[variable]
            for variable in range(3)
        ],
    )
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
        rtol=0.01 if dtype == torch.bfloat16 else 2e-4,
        atol=2e-5 if dtype == torch.bfloat16 else 2e-6,
    )
    assert torch.autograd.grad(actual.sum(), target, allow_unused=True)[0] is None


def test_field_gathering_and_fp32_casting_are_chunk_local(monkeypatch: pytest.MonkeyPatch) -> None:
    pred, target = _data((10, 9), batch=2, time=2, ensemble=2, variables=3)
    pred = pred.bfloat16().transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    target = target[:, :, :1].bfloat16()
    loss = WaveletAMSELoss([7, 9], J=2, subgrid=(5, 68), transform_field_chunk_size=5)
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
    expected = WaveletAMSELoss([7, 9], J=2)(pred[:, :, :, 5:68].float(), target[:, :, :, 5:68].float())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])
    assert chunk_sizes == [5, 5, 5, 5, 5, 5, 5, 5, 4, 4]


def test_power_and_cross_terms_are_summed_before_amse() -> None:
    loss = WaveletAMSELoss([2, 2], J=1)
    pred = torch.zeros(1, 1, 6, 1, 2, 2)
    target = torch.zeros_like(pred)
    pred[0, 0, 0, 0, 0, 0] = 1
    target[0, 0, 0, 0, 1, 0] = 1
    lowpass = torch.zeros(1, 1, 2, 2)
    actual = loss._score_pyramid(
        WaveletPyramid(lowpass, (pred,)),
        WaveletPyramid(lowpass, (target,)),
    )
    torch.testing.assert_close(actual, torch.tensor([0.5]))


def test_ignore_nans_masks_affected_coefficients_and_preserves_gradients() -> None:
    pred, target = _data(variables=1)
    ordinary_loss = WaveletAMSELoss([8, 10], J=2)
    nan_aware_loss = WaveletAMSELoss([8, 10], J=2, ignore_nans=True)
    torch.testing.assert_close(nan_aware_loss(pred, target), ordinary_loss(pred, target))

    missing_location = (0, 0, 0, 35, 0)
    pred[missing_location] = float("nan")
    pred.requires_grad_()
    score = nan_aware_loss(pred, target)
    assert torch.isfinite(score)
    pred_gradient = torch.autograd.grad(score, pred)[0]
    assert torch.isfinite(pred_gradient).all()
    assert pred_gradient[missing_location] == 0


def test_ignore_nans_masks_target_and_all_missing_fields() -> None:
    pred, target = _data(variables=1)
    target[..., 0, :] = float("nan")
    loss = WaveletAMSELoss([8, 10], J=2, ignore_nans=True)
    assert torch.isfinite(loss(pred, target))
    all_missing_score = loss(torch.full_like(pred, float("nan")), torch.full_like(target, float("nan")))
    torch.testing.assert_close(all_missing_score, torch.zeros_like(all_missing_score))


@pytest.mark.parametrize("field", ["pred", "target"])
def test_ignore_nans_does_not_mask_infinities(field: str) -> None:
    pred, target = _data()
    (pred if field == "pred" else target)[..., 1, 0] = float("inf")
    loss = WaveletAMSELoss([8, 10], J=2, ignore_nans=True)
    assert not torch.isfinite(loss(pred, target))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["pred", "target"])
def test_nonfinite_inputs_propagate_without_projection(bad: float, field: str) -> None:
    pred, target = _data()
    (pred if field == "pred" else target)[..., 1, 0] = bad
    loss = WaveletAMSELoss([8, 10], J=2)
    assert not torch.isfinite(loss(pred, target))


def test_lowpass_only_constant_field_and_finite_difference_gradient() -> None:
    loss = WaveletAMSELoss([8, 10], J=1, scale_weights=[0])
    pred = torch.ones(1, 1, 1, 80, 1, requires_grad=True)
    target = torch.zeros_like(pred)
    torch.testing.assert_close(loss(pred, target), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
    pred, target = _data(variables=1)
    pred.requires_grad_()
    loss = WaveletAMSELoss([8, 10], J=2)
    direction = torch.randn_like(pred)
    derivative = (torch.autograd.grad(loss(pred, target), pred)[0] * direction).sum()
    step = 1e-3
    finite_difference = (loss(pred + step * direction, target) - loss(pred - step * direction, target)) / (2 * step)
    torch.testing.assert_close(derivative, finite_difference, rtol=0.01, atol=3e-4)


@pytest.mark.parametrize(
    ("transform_separately", "field_chunk_size"),
    [(False, None), (True, None), (False, 3), (True, 1)],
)
def test_chunking_and_target_detachment(transform_separately: bool, field_chunk_size: int | None) -> None:
    pred, target = _data(batch=2, time=2, ensemble=3)
    target = target.clone().requires_grad_()
    pred.requires_grad_()
    baseline = WaveletAMSELoss([8, 10], J=2)(pred, target)
    actual = WaveletAMSELoss(
        [8, 10],
        J=2,
        transform_prediction_and_target_separately=transform_separately,
        transform_field_chunk_size=field_chunk_size,
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
    loss = WaveletAMSELoss([8, 10], J=2, subgrid=(5, 85))
    reference = WaveletAMSELoss([8, 10], J=2)
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


def test_nonspatial_scalers_apply_to_collapsed_scores_and_support_exclusions() -> None:
    pred, target = _data(time=2, variables=3)
    baseline_loss = WaveletAMSELoss([8, 10], J=2)
    baseline_by_time = torch.stack(
        [baseline_loss(pred[:, index : index + 1], target[:, index : index + 1], squash=False) for index in range(2)],
    )
    loss = WaveletAMSELoss([8, 10], J=2)
    time_weights = torch.tensor([0.25, 0.75])
    variable_weights = torch.tensor([1.0, 2.0, 3.0])
    loss.add_scaler(1, time_weights, name="time")
    loss.add_scaler(4, variable_weights, name="variable")

    weighted_time_baseline = (baseline_by_time * time_weights[:, None]).sum(dim=0)
    unweighted_time_baseline = baseline_by_time.sum(dim=0)
    torch.testing.assert_close(
        loss(pred, target, squash=False),
        weighted_time_baseline * variable_weights,
    )
    torch.testing.assert_close(
        loss(pred, target, squash=False, without_scalers=["variable"]),
        weighted_time_baseline,
    )
    torch.testing.assert_close(
        loss(pred, target, squash=False, without_scalers=[4]),
        weighted_time_baseline,
    )
    torch.testing.assert_close(
        loss(pred, target, squash=False, without_scalers=[1]),
        unweighted_time_baseline * variable_weights,
    )


def test_autocast_and_no_grad_contexts() -> None:
    pred, target = _data()
    pred.requires_grad_()
    loss = WaveletAMSELoss([8, 10], J=2)
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
    loss = WaveletAMSELoss([8, 10], J=2)
    out = loss(pred, target)
    assert torch.isfinite(out) and out >= -1e-6
    assert torch.isfinite(torch.autograd.grad(out, pred)[0]).all()
    assert out < 1e-5


def test_default_levels_and_transform_configuration() -> None:
    pred, target = _data((32, 48))
    automatic = WaveletAMSELoss([32, 48], transform={"biort": "near_sym_a", "qshift": "qshift_a"})
    explicit = WaveletAMSELoss([32, 48], J=2, transform={"biort": "near_sym_a", "qshift": "qshift_a"})
    assert automatic.J == 2
    torch.testing.assert_close(automatic(pred, target), explicit(pred, target))
    assert WaveletAMSELoss(
        [32, 48],
        separate_transform_gridpoint_threshold=100,
    ).transform_prediction_and_target_separately


def test_scaler_dimension_order_broadcasting_and_exclusions() -> None:
    pred, target = _data(variables=2)
    time_variable = torch.tensor([[1.0, 2.0]])
    reference = WaveletAMSELoss([8, 10], J=2)
    reordered = WaveletAMSELoss([8, 10], J=2)
    reference.add_scaler((1, 4), time_variable, name="mask")
    reordered.add_scaler((-1, -4), time_variable.T, name="mask")
    torch.testing.assert_close(reordered(pred, target), reference(pred, target))
    torch.testing.assert_close(
        reordered(pred, target, without_scalers=[4]),
        reference(pred, target, without_scalers=["mask"]),
    )
    with pytest.raises(ValueError, match="five-dimensional"):
        reordered.add_scaler(5, torch.ones(1), name="invalid_axis")
    reordered.add_scaler(0, torch.ones(3), name="invalid_batch")
    with pytest.raises(ValueError, match="does not broadcast"):
        reordered(pred, target)


def test_loss_variable_mapper_aligns_nonspatial_weights() -> None:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.utils.index_space import IndexSpace

    indices = IndexCollection(OmegaConf.create({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1, "c": 2})
    scalers = {
        "variable": (4, torch.tensor([1.0, 2.0, 3.0])),
    }
    mapped = get_loss_function(
        OmegaConf.create(
            {
                "_target_": "anemoi.training.losses.WaveletAMSELoss",
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
    direct = WaveletAMSELoss([8, 10], J=2)
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
    loss = WaveletAMSELoss([8, 10], J=2)
    model.add_module("loss", loss)
    config = OmegaConf.create(
        [{"module": "anemoi.training.losses.WaveletAMSELoss", "options": {"fullgraph": True, "dynamic": False}}],
    )
    mark_for_compilation(model, config)
    assert calls == [(loss._score_pyramid, {"fullgraph": True, "dynamic": False})]


@pytest.mark.parametrize(
    "options",
    [
        {"grid_shape": [1, 8]},
        {"J": 0},
        {"eps": 0},
        {"scale_weights": [1, -1]},
        {"scale_weights": [float("nan")]},
        {"scale_weights": [float("inf")]},
        {"subgrid": (5, 2)},
        {"transform_field_chunk_size": 0},
    ],
)
def test_invalid_configuration_is_rejected_by_schema(options: dict[str, Any]) -> None:
    config = {"_target_": "anemoi.training.losses.WaveletAMSELoss", "grid_shape": [8, 10], "J": 2, **options}
    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python(config)


def test_scale_weight_count_matches_wavelet_levels() -> None:
    with pytest.raises(ValueError, match="one weight per wavelet level"):
        WaveletAMSELoss([8, 10], J=2, scale_weights=[1])


@pytest.mark.parametrize("removed_mode", ["band_summed", "per_point"])
def test_mode_selector_is_not_supported(removed_mode: str) -> None:
    options: dict[str, Any] = {"grid_shape": [8, 10], "J": 2, "psd_mode": removed_mode}
    with pytest.raises(TypeError, match="psd_mode"):
        WaveletAMSELoss(**options)
    with pytest.raises(ValidationError, match="psd_mode"):
        TypeAdapter(LossSchemas).validate_python({"_target_": "anemoi.training.losses.WaveletAMSELoss", **options})


@pytest.mark.parametrize("dimension", [3, (1, 3), (0, 2, 3, 4)])
def test_area_scalers_are_rejected_on_add_and_update(dimension: int | tuple[int, ...]) -> None:
    loss = WaveletAMSELoss([8, 10], J=2)
    with pytest.raises(ValueError, match="does not support area weights"):
        loss.add_scaler(dimension, torch.ones(80), name="area")

    with pytest.raises(ValueError, match="does not support area weights"):
        get_loss_function(
            OmegaConf.create(
                {
                    "_target_": "anemoi.training.losses.WaveletAMSELoss",
                    "grid_shape": [8, 10],
                    "J": 2,
                    "scalers": ["area"],
                },
            ),
            {"area": (dimension, torch.ones(80))},
        )

    loss.scaler.add_scaler(dimension=3, scaler=torch.ones(80), name="legacy_area")
    with pytest.raises(ValueError, match="does not support area weights"):
        loss.update_scaler("legacy_area", torch.ones(80))
    with pytest.raises(ValueError, match="does not support area weights"):
        loss(*_data(), without_scalers=["legacy_area"])


def test_shard_metadata_and_grid_dimension_validation() -> None:
    pred, target = _data()
    loss = WaveletAMSELoss([8, 10], J=2)
    for kwargs in (
        {"grid_shard_slice": slice(0, 80)},
        {"grid_shard_sizes": [80]},
        {"grid_shard_slice": slice(0, 80), "grid_shard_sizes": [80]},
        {"grid_dim": 0},
        {"scaler_indices": [0]},
    ):
        with pytest.raises((ValueError, TypeError)):
            loss(pred, target, **kwargs)
    loss.add_scaler(4, torch.ones(3), name="bad_shape")
    with pytest.raises(ValueError, match="does not broadcast"):
        loss(pred, target)


def test_schema_factory_combination_and_output_mask_resolution() -> None:
    from anemoi.training.train.methods.base import resolve_subgrid

    config = {
        "_target_": "anemoi.training.losses.WaveletAMSELoss",
        "grid_shape": [8, 10],
        "J": 2,
        "subgrid": "output_mask",
        "transform": {"backend": "pytorch_wavelets"},
    }
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert isinstance(validated, WaveletAMSELossSchema)
    data_config = OmegaConf.create({"regional": config})

    class Mask:
        def as_tuple(self) -> tuple[int, int]:
            return (5, 85)

    resolve_subgrid(data_config, {"regional": Mask()})
    standalone = get_loss_function(data_config.regional)
    combined_config = OmegaConf.create(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [OmegaConf.to_container(data_config.regional)],
            "loss_weights": [0.4],
        },
    )
    TypeAdapter(LossSchemas).validate_python(OmegaConf.to_container(combined_config))
    combined = get_loss_function(combined_config)
    pred, target = _data((9, 10))
    torch.testing.assert_close(combined(pred, target), 0.4 * standalone(pred, target))


@pytest.mark.usefixtures("isolated_compile_cache")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_real_reference_compiled_score_forward_backward_and_scaler_update(backend: str) -> None:
    pred, target = _data((7, 9), ensemble=2)
    options = {"grid_shape": [7, 9], "J": 2, "transform_field_chunk_size": 2}
    eager, compiled = WaveletAMSELoss(**options), WaveletAMSELoss(**options)
    for loss in (eager, compiled):
        loss.add_scaler(4, torch.ones(2), name="variable")
    compiled.compile_for_training(backend=backend, fullgraph=True, dynamic=False)
    for i in range(2):
        variable_weights = torch.tensor([1.0, 1.0 + i])
        for loss in (eager, compiled):
            loss.update_scaler("variable", variable_weights)
        p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
        a, b = eager(p1, target), compiled(p2, target)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(torch.autograd.grad(a, p1)[0], torch.autograd.grad(b, p2)[0])


@pytest.mark.usefixtures("isolated_compile_cache")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_uniform_scoring_and_partial_chunks(backend: str) -> None:
    pred, target = _data((7, 9), ensemble=2)
    options = {"grid_shape": [7, 9], "J": 2, "transform_field_chunk_size": 3}
    eager, compiled = WaveletAMSELoss(**options), WaveletAMSELoss(**options)
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
    eager, compiled = WaveletAMSELoss(**options), WaveletAMSELoss(**options)
    compiled.compile_for_training(fullgraph=True, dynamic=False)
    p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    actual, expected = compiled(p1, target), eager(p2, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, p1)[0], torch.autograd.grad(expected, p2)[0])


def _distributed_worker(rank: int, rendezvous: str, variables: int) -> None:
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
        options = {
            "grid_shape": [6, 8],
            "J": 2,
            "subgrid": (3, 51),
            "transform_field_chunk_size": 1,
        }
        full_loss = WaveletAMSELoss(**options)
        loss = WaveletAMSELoss(**options)
        sizes = [19, 35]
        start = sum(sizes[:rank])
        shard_slice = slice(start, start + sizes[rank])
        for item in (full_loss, loss):
            item.add_scaler(4, torch.arange(1, variables + 1).float(), name="variable")
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
        variable_weights = torch.arange(1, variables + 1).float()
        loss.update_scaler("variable", variable_weights)
        full_loss.update_scaler("variable", variable_weights)
        torch.testing.assert_close(loss(pred, target, group=torch.distributed.group.WORLD), expected)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("variables", [1, 3])
def test_multiprocess_uneven_grid_and_variable_shards(
    tmp_path: Path,
    variables: int,
) -> None:
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), variables),
        nprocs=2,
        join=True,
    )


def _projection_matrix() -> torch.Tensor:
    """Return a known constant-preserving 13-source to 48-target interpolation."""
    matrix = torch.zeros(48, 13)
    rows = torch.arange(48)
    matrix[rows, rows % 13] = 0.25
    matrix[rows, (rows + 4) % 13] = 0.75
    return matrix


def _projection_graph(matrix: torch.Tensor) -> tuple[HeteroData, dict[str, Any]]:
    """Encode a small dense reference matrix as weighted source-to-target edges."""
    graph = HeteroData()
    graph["source"].num_nodes = matrix.shape[1]
    graph["wavelet_grid"].num_nodes = matrix.shape[0]
    rows, columns = matrix.nonzero(as_tuple=True)
    graph["source", "to", "wavelet_grid"].edge_index = torch.stack((columns, rows))
    graph["source", "to", "wavelet_grid"].weight = matrix[rows, columns]
    return graph, {
        "edges_name": ("source", "to", "wavelet_grid"),
        "edge_weight_attribute": "weight",
    }


def _dense_projection(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Independently project every field, retaining dense-input autograd."""
    return torch.einsum("gs,btesv->btegv", matrix, values.float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transform_separately", [False, True])
def test_projection_values_and_source_gradients_match_dense_reference(
    dtype: torch.dtype,
    transform_separately: bool,
) -> None:
    matrix = _projection_matrix()
    graph, config = _projection_graph(matrix)
    pred, target = _data((1, 13), batch=2, time=2, ensemble=2, variables=3)
    pred = pred.to(dtype).transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_()
    target = target[:, :, :1].to(dtype).requires_grad_()
    variables = torch.tensor([0.5, 2, 1.25])
    loss = WaveletAMSELoss(
        [6, 8],
        J=2,
        projection_config=config,
        graph_data=graph,
        scale_weights=[0.25, 1.5],
        transform_field_chunk_size=5,
        transform_prediction_and_target_separately=transform_separately,
    )
    loss.add_scaler(4, variables, name="variable")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss(pred, target, squash=False)
    expected = variables * _manual_reference(
        _dense_projection(pred, matrix),
        _dense_projection(target.expand_as(pred), matrix),
        (6, 8),
        2,
        [0.25, 1.5],
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
        rtol=0.01 if dtype == torch.bfloat16 else 2e-4,
        atol=2e-5 if dtype == torch.bfloat16 else 2e-6,
    )
    assert torch.autograd.grad(actual.sum(), target, allow_unused=True)[0] is None


def test_projection_ignore_nans_propagates_source_validity() -> None:
    matrix = _projection_matrix()
    graph, config = _projection_graph(matrix)
    pred, target = _data((1, 13), variables=1)
    pred[..., 0, 0] = float("nan")
    pred.requires_grad_()
    loss = WaveletAMSELoss(
        [6, 8],
        J=2,
        projection_config=config,
        graph_data=graph,
        ignore_nans=True,
        transform_field_chunk_size=2,
    )

    score = loss(pred, target, squash=False)
    assert torch.isfinite(score).all()
    pred_gradient = torch.autograd.grad(score.sum(), pred)[0]
    assert torch.isfinite(pred_gradient).all()
    assert pred_gradient[..., 0, 0].eq(0).all()


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("row_normalize", [False, True])
def test_projection_identity_and_permutation_preserve_target_order(permuted: bool, row_normalize: bool) -> None:
    matrix = torch.eye(48)
    if permuted:
        matrix = matrix.flip(0)
    graph, config = _projection_graph(matrix)
    loss = WaveletAMSELoss(
        [6, 8],
        J=2,
        projection_config={**config, "row_normalize": row_normalize},
        graph_data=graph,
    )
    pred, target = _data((6, 8))
    pred.requires_grad_()
    actual = loss(pred, target)
    expected = WaveletAMSELoss([6, 8], J=2)(
        pred.flip(3) if permuted else pred,
        target.flip(3) if permuted else target,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, pred)[0], torch.autograd.grad(expected, pred)[0])


def test_projection_knn_target_builder_schema_and_factory() -> None:
    graph = HeteroData()
    graph["source"].x = torch.deg2rad(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    graph["source"].num_nodes = 4
    config = {
        "_target_": "anemoi.training.losses.WaveletAMSELoss",
        "grid_shape": [2, 3],
        "J": 1,
        "scalers": ["variable"],
        "projection_config": {
            "target_node_name": "wavelet_grid",
            "num_nearest_neighbours": 2,
            "sigma": 0.02,
            "node_builder": {
                "_target_": "anemoi.graphs.nodes.LatLonNodes",
                "latitudes": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "longitudes": [0.25, 0.5, 0.75, 0.25, 0.5, 0.75],
            },
        },
    }
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert validated.projection_config.row_normalize
    loss = get_loss_function(
        OmegaConf.create(validated.model_dump(by_alias=True)),
        {"variable": (4, torch.tensor([2.0]))},
        graph_data=graph,
        data_node_name="source",
    )
    matrix = loss.projection_provider.get_edges().to_dense()
    torch.testing.assert_close(matrix.sum(1), torch.ones(6))
    torch.testing.assert_close(matrix[1], torch.tensor([0.5, 0.5, 0.0, 0.0]))
    torch.testing.assert_close(matrix[4], torch.tensor([0.0, 0.0, 0.5, 0.5]))
    torch.testing.assert_close(
        loss.projector(torch.ones(2, 4, 1), loss.projection_provider.get_edges()),
        torch.ones(2, 6, 1),
    )
    fields = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(1, 1, 1, 4, 1).requires_grad_()
    zeros = torch.zeros_like(fields)
    expected = 2 * _manual_reference(_dense_projection(fields, matrix), torch.zeros(1, 1, 1, 6, 1), (2, 3), 1, [1])
    actual = loss(fields, zeros, squash=False)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(torch.autograd.grad(actual.sum(), fields)[0]).all()
    assert graph.node_types == ["source"]
    combined_config = {
        "_target_": "anemoi.training.losses.combined.CombinedLoss",
        "losses": [config],
        "loss_weights": [0.4],
    }
    TypeAdapter(LossSchemas).validate_python(combined_config)
    combined = get_loss_function(
        OmegaConf.create(combined_config),
        {"variable": (4, torch.tensor([2.0]))},
        graph_data=graph,
        data_node_name="source",
    )
    torch.testing.assert_close(combined(fields, zeros), 0.4 * actual.mean())
    with pytest.raises(ValueError, match="n_neighbors"):
        WaveletAMSELoss(
            [2, 3],
            projection_config={**config["projection_config"], "num_nearest_neighbours": 5},
            graph_data=graph,
            data_node_name="source",
        )
    with pytest.raises(ValueError, match="missing from graph_data"):
        WaveletAMSELoss(
            [2, 3],
            projection_config=config["projection_config"],
            graph_data=graph,
            data_node_name="missing",
        )
    assert not SpectralProjectionConfigSchema(matrix_path="unused.npz").row_normalize


def test_projection_loss_variable_mapping_and_scaler_subsets() -> None:
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.utils.index_space import IndexSpace

    matrix = _projection_matrix()
    graph, config = _projection_graph(matrix)
    indices = IndexCollection(OmegaConf.create({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1, "c": 2})
    loss = get_loss_function(
        OmegaConf.create(
            {
                "_target_": "anemoi.training.losses.WaveletAMSELoss",
                "grid_shape": [6, 8],
                "J": 2,
                "projection_config": config,
                "predicted_variables": ["c", "a"],
                "scalers": ["variable"],
            },
        ),
        {"variable": (4, torch.tensor([1.0, 2.0, 3.0]))},
        data_indices=indices,
        graph_data=graph,
    )
    pred, target = _data((1, 13), variables=3)
    pred.requires_grad_()
    actual = loss(
        pred,
        target,
        squash=False,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.MODEL_OUTPUT,
    )
    expected = torch.tensor([3.0, 1.0]) * _manual_reference(
        _dense_projection(pred[..., [2, 0]], matrix),
        _dense_projection(target[..., [2, 0]], matrix),
        (6, 8),
        2,
        [1, 1],
    )
    expected_full = torch.zeros(3)
    expected_full[[2, 0]] = expected
    torch.testing.assert_close(actual, expected_full)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), pred)[0],
        torch.autograd.grad(expected.sum(), pred)[0],
    )


def test_projection_file_deepcopy_and_module_state(tmp_path: Path) -> None:
    from scipy.sparse import csr_matrix
    from scipy.sparse import save_npz

    matrix = _projection_matrix()
    matrix_path = tmp_path / "projection.npz"
    save_npz(matrix_path, csr_matrix((2 * matrix).numpy()))
    loss = WaveletAMSELoss(
        [6, 8],
        J=2,
        projection_config={"matrix_path": str(matrix_path)},
    )
    copied = copy.deepcopy(loss).to("cpu")
    assert copied.projection_provider.get_edges() is loss.projection_provider.get_edges()
    assert not hasattr(loss, "target_grid_weights")
    pred, target = _data((1, 13))
    expected = _manual_reference(
        _dense_projection(pred, matrix),
        _dense_projection(target, matrix),
        (6, 8),
        2,
        [1, 1],
    )
    torch.testing.assert_close(copied(pred, target, squash=False), expected)
    torch.testing.assert_close(copied(pred, target), loss(pred, target))


def test_projection_and_unprojected_losses_reject_area_scalers() -> None:
    graph, config = _projection_graph(torch.eye(48))
    for options in ({}, {"projection_config": config, "graph_data": graph}):
        loss = WaveletAMSELoss([6, 8], J=2, **options)
        with pytest.raises(ValueError, match="does not support area weights"):
            loss.add_scaler(3, torch.ones(48), name="area")


def test_projection_gathering_and_projection_stay_chunk_local(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, config = _projection_graph(_projection_matrix())
    loss = WaveletAMSELoss(
        [6, 8],
        J=2,
        projection_config=config,
        graph_data=graph,
        transform_field_chunk_size=5,
    )
    pred, target = (t.bfloat16() for t in _data((1, 13), batch=2, time=2, ensemble=2, variables=3))
    project = loss.projector.forward
    gather = loss._flat_fields
    chunks = []

    def checked_gather(values: torch.Tensor, indices: tuple[torch.Tensor, ...]) -> torch.Tensor:
        assert indices[0].numel() <= 5
        fields = gather(values, indices)
        assert fields.shape == (indices[0].numel(), values.shape[3])
        assert fields.dtype == torch.float32
        return fields

    def checked_project(fields: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        assert fields.shape[0] <= 5
        assert fields.shape[1:] == (13, 1)
        assert fields.dtype == matrix.dtype == torch.float32
        assert matrix.layout == torch.sparse_csr
        chunks.append(fields.shape[0])
        return project(fields, matrix)

    def no_dense(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Projection must not densify its sparse matrix")

    monkeypatch.setattr(loss.projector, "forward", checked_project)
    monkeypatch.setattr(loss, "_flat_fields", checked_gather)
    monkeypatch.setattr(torch.Tensor, "to_dense", no_dense)
    assert torch.isfinite(loss(pred, target))
    assert chunks == [5, 5] * 4 + [4, 4]


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("negative", "finite and nonnegative"),
        ("nan", "finite and nonnegative"),
        ("inf", "finite and nonnegative"),
        ("empty_row", "positive support"),
        ("destination_shape", "destination point count"),
        ("unnormalized", "rows summing to one"),
    ],
)
def test_projection_invalid_matrices(kind: str, message: str) -> None:
    matrix = _projection_matrix()
    options = {}
    if kind == "negative":
        matrix = -matrix
    elif kind in ("nan", "inf"):
        matrix[0, 0] = float(kind)
    elif kind == "empty_row":
        matrix[0] = 0
    elif kind == "destination_shape":
        matrix = matrix[:-1]
    elif kind == "unnormalized":
        matrix = 2 * matrix
        options["row_normalize"] = False
    graph, config = _projection_graph(matrix)
    with pytest.raises(ValueError, match=message):
        WaveletAMSELoss([6, 8], J=2, projection_config={**config, **options}, graph_data=graph)


@pytest.mark.parametrize(
    "options",
    [
        {"projection_config": {}},
        {"projection_config": {"matrix_path": "unused.npz", "edges_name": ("a", "to", "b")}},
        {"projection_config": {"matrix_path": "unused.npz", "sigma": 1}},
        {"projection_config": {"num_nearest_neighbours": 0, "sigma": 1, "grid": "O16"}},
        {"projection_config": {"num_nearest_neighbours": 1, "sigma": 0, "grid": "O16"}},
        {"projection_config": {"num_nearest_neighbours": 1, "sigma": float("inf"), "grid": "O16"}},
        {"projection_config": {"matrix_path": "unused.npz"}, "subgrid": (0, None)},
    ],
)
def test_projection_schema_rejects_invalid_options(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python(
            {"_target_": "anemoi.training.losses.WaveletAMSELoss", "grid_shape": [6, 8], **options},
        )


def test_ignore_nans_is_accepted_by_constructor_and_schema() -> None:
    config = {
        "_target_": "anemoi.training.losses.WaveletAMSELoss",
        "grid_shape": [6, 8],
        "ignore_nans": True,
    }
    assert WaveletAMSELoss([6, 8], ignore_nans=True).ignore_nans
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert isinstance(validated, WaveletAMSELossSchema)
    assert validated.ignore_nans


def test_target_grid_weights_is_not_a_loss_option() -> None:
    config = {"_target_": "anemoi.training.losses.WaveletAMSELoss", "grid_shape": [6, 8]}
    with pytest.raises(TypeError, match="target_grid_weights"):
        WaveletAMSELoss([6, 8], target_grid_weights=[1.0] * 48)
    with pytest.raises(ValidationError, match="target_grid_weights"):
        TypeAdapter(LossSchemas).validate_python({**config, "target_grid_weights": [1.0] * 48})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["pred", "target"])
def test_projection_propagates_nonfinite_source_values(bad: float, field: str) -> None:
    graph, config = _projection_graph(_projection_matrix())
    loss = WaveletAMSELoss(
        [6, 8],
        projection_config=config,
        graph_data=graph,
        transform_field_chunk_size=1,
    )
    pred, target = _data((1, 13))
    (pred if field == "pred" else target)[..., 1, 0] = bad
    assert not torch.isfinite(loss(pred, target))


def test_projection_missing_context(tmp_path: Path) -> None:
    _, config = _projection_graph(_projection_matrix())
    with pytest.raises(ValueError, match="graph_data"):
        WaveletAMSELoss([6, 8], projection_config=config)
    with pytest.raises(FileNotFoundError):
        WaveletAMSELoss([6, 8], projection_config={"matrix_path": str(tmp_path / "missing.npz")})


@pytest.mark.usefixtures("isolated_compile_cache")
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_projection_real_compiled_scoring(backend: str) -> None:
    graph, config = _projection_graph(_projection_matrix())
    options = {
        "grid_shape": [6, 8],
        "J": 2,
        "projection_config": config,
        "graph_data": graph,
        "transform_field_chunk_size": 2,
    }
    eager, compiled = WaveletAMSELoss(**options), WaveletAMSELoss(**options)
    compiled.compile_for_training(backend=backend, fullgraph=True, dynamic=False)
    pred, target = _data((1, 13), variables=3)
    p1, p2 = pred.clone().requires_grad_(), pred.clone().requires_grad_()
    actual, expected = compiled(p1, target), eager(p2, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, p1)[0], torch.autograd.grad(expected, p2)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU sparse projection requires a GPU")
def test_projection_gpu_matches_cpu_values_and_gradients() -> None:
    graph, config = _projection_graph(_projection_matrix())
    loss = WaveletAMSELoss(
        [6, 8],
        J=2,
        projection_config=config,
        graph_data=graph,
        transform_field_chunk_size=1,
    )
    pred, target = _data((1, 13))
    pred.requires_grad_()
    expected = loss(pred, target)
    expected_gradient = torch.autograd.grad(expected, pred)[0]
    device_pred = pred.detach().cuda().requires_grad_()
    actual = loss.cuda()(device_pred, target.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, device_pred)[0].cpu(),
        expected_gradient,
        rtol=2e-4,
        atol=2e-6,
    )


def _distributed_projection_worker(rank: int, rendezvous: str, variables: int, nonfinite: bool) -> None:
    """Compare sharded projection scoring, including nonfinite inputs on one rank."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=60),
    )
    try:
        matrix = _projection_matrix()
        graph, config = _projection_graph(matrix)
        loss = WaveletAMSELoss(
            [6, 8],
            J=2,
            projection_config=config,
            graph_data=graph,
            transform_field_chunk_size=1,
        )
        loss.add_scaler(4, torch.arange(1, variables + 1).float(), name="variable")
        pred, target = _data((1, 13), variables=variables, ensemble=2)
        sizes = [5, 8]
        start = sum(sizes[:rank])
        shard_slice = slice(start, start + sizes[rank])
        local_pred = pred[:, :, :, shard_slice].clone().requires_grad_()
        local_target = target[:, :, :, shard_slice].clone()
        if nonfinite and rank == 0:
            local_target[..., 0, -1] = float("nan")
        kwargs = {
            "grid_shard_slice": shard_slice,
            "grid_shard_sizes": sizes,
            "group": torch.distributed.group.WORLD,
        }
        if nonfinite:
            actual = loss(local_pred, local_target, **kwargs)
            assert not torch.isfinite(actual).all()
            torch.distributed.barrier()
        else:
            full_pred = pred.clone().requires_grad_()
            expected = _manual_reference(
                _dense_projection(full_pred, matrix),
                _dense_projection(target, matrix),
                (6, 8),
                2,
                [1, 1],
            )
            expected = (expected * torch.arange(1, variables + 1)).mean()
            expected.backward()
            actual = loss(local_pred, local_target, **kwargs)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            torch.testing.assert_close(local_pred.grad, full_pred.grad[:, :, :, shard_slice], atol=2e-6, rtol=2e-5)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize(("variables", "nonfinite"), [(3, False), (1, False), (3, True), (1, True)])
def test_projection_multiprocess_unequal_grids_with_nonfinite_inputs(
    tmp_path: Path,
    variables: int,
    nonfinite: bool,
) -> None:
    torch.multiprocessing.spawn(
        _distributed_projection_worker,
        args=(str(tmp_path / "projection-rendezvous"), variables, nonfinite),
        nprocs=2,
        join=True,
    )
