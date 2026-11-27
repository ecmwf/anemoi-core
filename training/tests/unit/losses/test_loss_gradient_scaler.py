# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the loss-gradient rescaling of arXiv:2306.06079, section 4.3.2 (`LossScaler`)."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.data import GriddedSource
from anemoi.models.data import TabularSource
from anemoi.models.data import TensorLayout
from anemoi.training.losses import LossScaler
from anemoi.training.losses import scaler_tensor
from anemoi.training.losses.scaler_tensor import LossScaleFunction

SHAPE = (2, 3, 2, 7, 4)  # gridded: (batch, time, ensemble, grid, variable)
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TABULAR_LAYOUTS = [TensorLayout(ensemble=0, grid=1, variables=2), TensorLayout(grid=0, variables=1)]
VARIABLE_WEIGHTS = torch.tensor([1.0, 10.0, 100.0, 0.1], dtype=torch.float64)


def _upstream(seed: int = 0, shape: tuple[int, ...] = SHAPE, variables_axis: int = -1) -> torch.Tensor:
    """Gradient of the loss w.r.t. the prediction, with very different scales per variable."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64).movedim(variables_axis, -1)
    return (values * VARIABLE_WEIGHTS[: values.shape[-1]]).movedim(-1, variables_axis)


def _gridded(values: torch.Tensor, layout: TensorLayout = GRIDDED_LAYOUT) -> GriddedSource:
    return GriddedSource(
        name="data",
        values=values,
        variables=[f"v{i}" for i in range(values.shape[layout.variables])],
        coordinates=torch.zeros(values.shape[layout.grid], 2),
        layout=layout,
    )


def _tabular(values: list[torch.Tensor], layout: TensorLayout) -> TabularSource:
    grid = layout.grid
    return TabularSource(
        name="obs",
        values=values,
        variables=[f"v{i}" for i in range(values[0].shape[layout.variables])],
        coordinates=[torch.zeros(v.shape[grid], 2) for v in values],
        layout=layout,
        timedeltas=[torch.zeros(v.shape[grid]) for v in values],
        boundaries=[(slice(0, v.shape[grid]),) for v in values],
    )


def _row_norms(grad: torch.Tensor) -> torch.Tensor:
    """Per-variable norms over (ensemble, grid) of a gridded gradient, shape (batch, time, variable)."""
    return grad.square().sum(dim=(GRIDDED_LAYOUT.ensemble, GRIDDED_LAYOUT.grid)).sqrt()


def _sample_norms(grad: torch.Tensor, layout: TensorLayout) -> torch.Tensor:
    """Per-variable norms of one tabular sample over all of its other axes, shape (variable,)."""
    return grad.movedim(layout.variables, -1).reshape(-1, grad.shape[layout.variables]).norm(dim=0)


def _harmonic_mean(norms: torch.Tensor) -> torch.Tensor:
    return norms.shape[-1] / norms.reciprocal().sum(dim=-1, keepdim=True)


def _scaled_grad(
    upstream: torch.Tensor,
    scaler: LossScaler | None = None,
    layout: TensorLayout = GRIDDED_LAYOUT,
    **kwargs,
) -> torch.Tensor:
    """Gradient that reaches a gridded prediction when the loss is ``sum(upstream * scaler(x))``."""
    scaler = scaler or LossScaler(**kwargs)
    x = torch.zeros_like(upstream, requires_grad=True)
    (scaler(_gridded(x, layout)).values * upstream).sum().backward()
    return x.grad


def _scaled_tabular_grads(upstreams: list[torch.Tensor], layout: TensorLayout, **kwargs) -> list[torch.Tensor]:
    """Gradients that reach each sample of a tabular prediction when the loss is ``sum_i sum(upstream_i * x_i)``."""
    xs = [torch.zeros_like(u, requires_grad=True) for u in upstreams]
    out = LossScaler(**kwargs)(_tabular(xs, layout))
    sum((v * u).sum() for v, u in zip(out.values, upstreams, strict=True)).backward()
    return [x.grad for x in xs]


# ── gridded: forward / backward maths ───────────────────────────────────────


def test_forward_is_identity() -> None:
    x = torch.randn(SHAPE, requires_grad=True)
    source = _gridded(x)
    out = LossScaler()(source)

    assert isinstance(out, GriddedSource)
    assert torch.equal(out.values, x)
    assert out.values.requires_grad
    assert out.variables == source.variables
    assert out.layout == source.layout


def test_power_one_equalises_norms_per_row() -> None:
    upstream = _upstream()
    grad = _scaled_grad(upstream)

    norms = _row_norms(grad)
    # Every variable ends up with the same norm in each (batch, time) row...
    torch.testing.assert_close(norms, norms[..., :1].expand_as(norms))
    # ...namely the harmonic mean of the unscaled norms, since the weights average to 1.
    torch.testing.assert_close(norms, _harmonic_mean(_row_norms(upstream)).expand_as(norms))


def test_power_zero_is_identity() -> None:
    upstream = _upstream()
    torch.testing.assert_close(_scaled_grad(upstream, power=0.0), upstream)


def test_intermediate_power_reduces_spread() -> None:
    upstream = _upstream()
    norms_before = _row_norms(upstream)
    norms_after = _row_norms(_scaled_grad(upstream, power=0.5))
    # With power p the ratio of norms becomes ratio ** (1 - p).
    torch.testing.assert_close(
        norms_after.amax(-1) / norms_after.amin(-1),
        (norms_before.amax(-1) / norms_before.amin(-1)).sqrt(),
    )


@pytest.mark.parametrize("factor", [1e-6, 3.0, 1e6])
def test_rescaling_is_positively_homogeneous(factor: float) -> None:
    upstream = _upstream()
    torch.testing.assert_close(_scaled_grad(upstream * factor), factor * _scaled_grad(upstream))


def test_inactive_variable_stays_zero_and_is_excluded() -> None:
    upstream = _upstream()
    upstream[..., 1] = 0.0
    grad = _scaled_grad(upstream)

    assert torch.isfinite(grad).all()
    assert torch.equal(grad[..., 1], torch.zeros_like(grad[..., 1]))

    active = [0, 2, 3]
    norms = _row_norms(grad)[..., active]
    expected = _harmonic_mean(_row_norms(upstream)[..., active])
    torch.testing.assert_close(norms, expected.expand_as(norms))


def test_all_zero_gradient_row_is_finite_zero() -> None:
    upstream = _upstream()
    upstream[1, 2] = 0.0
    grad = _scaled_grad(upstream)

    assert torch.isfinite(grad).all()
    assert torch.equal(grad[1, 2], torch.zeros_like(grad[1, 2]))


def test_zero_gradient_everywhere_is_finite_zero() -> None:
    grad = _scaled_grad(torch.zeros(SHAPE, dtype=torch.float64))
    assert torch.equal(grad, torch.zeros_like(grad))


@pytest.mark.parametrize("max_gain", [1.0, 2.0, 50.0])
def test_max_gain_bounds_weight_ratio(max_gain: float) -> None:
    upstream = _upstream()
    grad = _scaled_grad(upstream, max_gain=max_gain)

    weights = _row_norms(grad) / _row_norms(upstream)
    ratio = weights.amax(-1) / weights.amin(-1)
    assert (ratio <= max_gain * (1 + 1e-9)).all()
    # The weights still average to one over the active variables.
    torch.testing.assert_close(weights.mean(-1), torch.ones_like(ratio))


def test_max_gain_of_one_is_identity() -> None:
    upstream = _upstream()
    torch.testing.assert_close(_scaled_grad(upstream, max_gain=1.0), upstream)


def test_rows_are_rescaled_independently() -> None:
    upstream = _upstream()
    other = upstream.clone()
    other[1] *= torch.tensor([1e3, 1.0, 1e-3, 5.0], dtype=torch.float64)
    other[0, 1:] = 0.0

    torch.testing.assert_close(_scaled_grad(other)[0, 0], _scaled_grad(upstream)[0, 0])


def test_low_precision_gradient_keeps_dtype() -> None:
    upstream = _upstream().to(torch.bfloat16)
    x = torch.zeros(SHAPE, dtype=torch.bfloat16, requires_grad=True)
    (LossScaler()(_gridded(x)).values * upstream).sum().backward()

    assert x.grad.dtype == torch.bfloat16
    reference = _scaled_grad(upstream.to(torch.float32))
    torch.testing.assert_close(x.grad.float(), reference, rtol=1e-2, atol=1e-2)


def test_scaled_output_composes_with_further_ops() -> None:
    """The view returned by the forward pass carries the rescaling through later ops (e.g. dtype casts)."""
    upstream = _upstream().float()
    x = torch.zeros(SHAPE, requires_grad=True)
    (LossScaler()(_gridded(x)).values.double() * upstream.double()).sum().backward()
    torch.testing.assert_close(x.grad, _scaled_grad(upstream))


def test_gridded_layout_with_variables_before_grid() -> None:
    upstream = _upstream()
    layout = TensorLayout(batch=0, time=1, ensemble=2, variables=3, grid=4)

    grad = _scaled_grad(upstream.movedim(4, 3), layout=layout)

    torch.testing.assert_close(grad, _scaled_grad(upstream).movedim(4, 3))


def test_gridded_layout_without_ensemble_axis() -> None:
    upstream = _upstream()[:, :, 0]  # (batch, time, grid, variable)
    grad = _scaled_grad(upstream, layout=TensorLayout(batch=0, time=1, grid=2, variables=3))

    norms = grad.square().sum(dim=2).sqrt()
    expected = _harmonic_mean(upstream.square().sum(dim=2).sqrt())
    torch.testing.assert_close(norms, expected.expand_as(norms))


# ── tabular ─────────────────────────────────────────────────────────────────


def _tabular_upstreams(layout: TensorLayout, sizes: tuple[int, ...] = (5, 1, 9), n_var: int = 4) -> list[torch.Tensor]:
    """One upstream gradient per sample, laid out per ``layout`` (2 ensemble members when it has the axis)."""
    upstreams = []
    for seed, size in enumerate(sizes):
        axes = {"ensemble": 2, "grid": size, "variables": n_var}
        shape = tuple(axes[name] for name in layout.axis_names)
        upstreams.append(_upstream(seed, shape, variables_axis=layout.variables))
    return upstreams


@pytest.mark.parametrize("layout", TABULAR_LAYOUTS, ids=["with-ensemble", "without-ensemble"])
def test_tabular_source_keeps_its_structure(layout: TensorLayout) -> None:
    xs = [torch.zeros_like(u, requires_grad=True) for u in _tabular_upstreams(layout)]
    source = _tabular(xs, layout)

    out = LossScaler()(source)

    assert isinstance(out, TabularSource)
    assert out.layout == source.layout
    assert len(out.values) == len(xs)
    for value, x in zip(out.values, xs, strict=True):
        assert torch.equal(value, x)
        assert value.requires_grad


@pytest.mark.parametrize("layout", TABULAR_LAYOUTS, ids=["with-ensemble", "without-ensemble"])
def test_tabular_source_equalises_norms_per_sample(layout: TensorLayout) -> None:
    """Each sample is one row: its variables end up with the harmonic mean of its own unscaled norms."""
    upstreams = _tabular_upstreams(layout)
    grads = _scaled_tabular_grads(upstreams, layout)

    for grad, upstream in zip(grads, upstreams, strict=True):
        norms = _sample_norms(grad, layout)
        torch.testing.assert_close(norms, _harmonic_mean(_sample_norms(upstream, layout)).expand_as(norms))


def test_tabular_samples_are_rescaled_independently() -> None:
    layout = TABULAR_LAYOUTS[0]
    upstreams = _tabular_upstreams(layout)
    other = [u.clone() for u in upstreams]
    other[1] *= 1e4
    other[2][..., 0] = 0.0

    torch.testing.assert_close(
        _scaled_tabular_grads(other, layout)[0],
        _scaled_tabular_grads(upstreams, layout)[0],
    )


def test_tabular_power_zero_is_identity() -> None:
    layout = TABULAR_LAYOUTS[1]
    upstreams = _tabular_upstreams(layout)
    for grad, upstream in zip(_scaled_tabular_grads(upstreams, layout, power=0.0), upstreams, strict=True):
        torch.testing.assert_close(grad, upstream)


def test_tabular_masked_variable_stays_zero() -> None:
    """A variable with no observations in a sample (zero gradient) is left out of that sample's normalisation."""
    layout = TABULAR_LAYOUTS[1]
    upstreams = _tabular_upstreams(layout)
    upstreams[0][:, 2] = 0.0
    grads = _scaled_tabular_grads(upstreams, layout)

    assert torch.equal(grads[0][:, 2], torch.zeros_like(grads[0][:, 2]))
    active = [0, 1, 3]
    norms = _sample_norms(grads[0], layout)[active]
    torch.testing.assert_close(norms, _harmonic_mean(_sample_norms(upstreams[0], layout)[active]).expand_as(norms))


def test_tabular_source_with_unused_sample() -> None:
    """A sample that the loss does not reach gets a zero gradient, without NaNs."""
    layout = TABULAR_LAYOUTS[1]
    upstream = _upstream(0, (6, 3))
    xs = [torch.zeros(6, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]

    out = LossScaler()(_tabular(xs, layout))
    (out.values[0] * upstream).sum().backward()

    assert torch.equal(xs[1].grad, torch.zeros_like(xs[1]))
    assert torch.isfinite(xs[0].grad).all()


# ── distributed reduction ───────────────────────────────────────────────────


def _forbid_all_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs) -> None:
        msg = "all_reduce must not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(scaler_tensor.dist, "all_reduce", fail)


def test_no_group_does_not_communicate(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_all_reduce(monkeypatch)
    _scaled_grad(_upstream())


def test_single_rank_group_does_not_communicate(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_all_reduce(monkeypatch)
    monkeypatch.setattr(scaler_tensor.dist, "get_world_size", lambda group=None: 1)  # noqa: ARG005

    x = torch.zeros(SHAPE, dtype=torch.float64, requires_grad=True)
    (LossScaler()(_gridded(x), group=object()).values * _upstream()).sum().backward()


def test_tabular_samples_share_one_all_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    """All samples go through a single collective; a replicated reduction (factor 2) changes nothing."""
    calls = []

    def all_reduce(tensor: torch.Tensor, group: object = None) -> None:
        calls.append((tensor.shape, group))
        tensor.mul_(2.0)  # what a sum over two ranks holding the same data would give

    monkeypatch.setattr(scaler_tensor.dist, "get_world_size", lambda group=None: 2)  # noqa: ARG005
    monkeypatch.setattr(scaler_tensor.dist, "all_reduce", all_reduce)

    layout = TABULAR_LAYOUTS[0]
    upstreams = _tabular_upstreams(layout, sizes=(3, 4, 5))
    group = object()

    def grads(group: object) -> list[torch.Tensor]:
        xs = [torch.zeros_like(u, requires_grad=True) for u in upstreams]
        out = LossScaler()(_tabular(xs, layout), group=group)
        sum((v * u).sum() for v, u in zip(out.values, upstreams, strict=True)).backward()
        return [x.grad for x in xs]

    with_group = grads(group)
    assert calls == [(torch.Size([3, 4]), group)]
    for got, expected in zip(with_group, grads(None), strict=True):
        torch.testing.assert_close(got, expected)


# ── construction ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, None),
        (False, None),
        (True, (1.0, None)),
        ({"enabled": False, "power": 0.5}, None),
        ({"enabled": True}, (1.0, None)),
        ({"enabled": True, "power": 0.5, "max_gain": 10.0}, (0.5, 10.0)),
        (OmegaConf.create({"enabled": True, "power": 0.25, "max_gain": None}), (0.25, None)),
        (SimpleNamespace(enabled=True, power=2.0, max_gain=3.0), (2.0, 3.0)),
        (SimpleNamespace(enabled=False), None),
    ],
)
def test_from_config(config: object, expected: tuple[float, float | None] | None) -> None:
    scaler = LossScaler.from_config(config)
    if expected is None:
        assert scaler is None
    else:
        assert isinstance(scaler, LossScaler)
        assert (scaler.power, scaler.max_gain) == expected


def test_scaler_is_stateless() -> None:
    scaler = LossScaler(power=0.5, max_gain=4.0)
    assert scaler.state_dict() == {}
    assert list(scaler.parameters()) == []
    assert repr(scaler) == "LossScaler(power=0.5, max_gain=4.0)"


@pytest.mark.parametrize(("kwargs", "match"), [({"power": -1.0}, "power"), ({"max_gain": 0.5}, "max_gain")])
def test_invalid_options_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LossScaler(**kwargs)


def test_plain_tensor_is_rejected() -> None:
    """A bare tensor has no layout, so the scaler cannot tell which axes to reduce over."""
    with pytest.raises(TypeError, match="expects a Source"):
        LossScaler()(torch.zeros(SHAPE, requires_grad=True))


def test_function_rejects_reducing_the_variable_dimension() -> None:
    with pytest.raises(ValueError, match="variable dimension"):
        LossScaleFunction.apply((1,), 1, 1.0, None, None, torch.zeros(2, 3, requires_grad=True))


def test_function_rejects_mismatched_segments() -> None:
    with pytest.raises(ValueError, match="equal rank and variable width"):
        LossScaleFunction.apply((0,), 1, 1.0, None, None, torch.zeros(2, 3), torch.zeros(2, 4))
