# (C) Copyright 2026- Anemoi contributors.
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
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import SpectralIncrementSmoothnessLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.increment_smoothness import EARTH_RADIUS_KM
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.da_mixin import DATrainingMixin
from anemoi.training.utils.index_space import IndexSpace

N48 = 10944  # O48 grid points (nlat 96)


def _octahedral_coords(grid: str) -> tuple[torch.Tensor, torch.Tensor]:
    from anemoi.graphs.nodes import ReducedGaussianGridNodes

    x = ReducedGaussianGridNodes(grid, "data").update_graph(HeteroData())["data"].x.double()
    return x[:, 0], x[:, 1]


@pytest.fixture(scope="module")
def o48() -> tuple[torch.Tensor, torch.Tensor]:
    return _octahedral_coords("o48")


def _loss(nlat: int = 96, **kwargs) -> SpectralIncrementSmoothnessLoss:
    return SpectralIncrementSmoothnessLoss(transform="octahedral_sht", nlat=nlat, **kwargs)


def _field(values: torch.Tensor) -> torch.Tensor:
    return values.float().view(1, 1, 1, -1, 1)


# ── operator ──────────────────────────────────────────────────────────────


def test_parseval_laplacian_of_degree_one_harmonic(o48: tuple[torch.Tensor, torch.Tensor]) -> None:
    """sin(lat) has nabla^2 = -2/R^2: penalty = (2 (ls/R)^2)^2 * mean(sin^2) = (2 (ls/R)^2)^2 / 3."""
    lat, _ = o48
    loss = _loss(order=1, length_scale_km=100.0)
    increment = _field(torch.sin(lat))
    got = float(loss(increment, increment, reference=torch.zeros_like(increment)))
    want = (2 * (100.0 / EARTH_RADIUS_KM) ** 2) ** 2 / 3
    assert got == pytest.approx(want, rel=1e-3)


def test_zero_for_constant_increment(o48: tuple[torch.Tensor, torch.Tensor]) -> None:
    lat, lon = o48
    loss = _loss()
    reference = _field(torch.cos(3 * lon) * torch.cos(lat))
    pred = reference + 5.0  # spatially constant increment (l = 0)
    assert float(loss(pred, pred, reference=reference)) == pytest.approx(0.0, abs=1e-10)


def test_ignores_full_field_structure() -> None:
    """A rough base with a zero increment costs nothing (unlike the full-field Laplacian loss)."""
    loss = _loss()
    reference = torch.randn(1, 1, 1, N48, 2)
    assert float(loss(reference.clone(), reference, reference=reference)) == 0.0


def test_penalty_is_resolution_independent() -> None:
    """The same smooth increment costs the same on O48 and O96."""
    penalties = []
    for grid, nlat in (("o48", 96), ("o96", 192)):
        lat, lon = _octahedral_coords(grid)
        increment = _field(torch.cos(lat) ** 4 * torch.sin(lat) * torch.cos(4 * lon))
        loss = _loss(nlat=nlat, order=2)
        penalties.append(float(loss(increment, increment, reference=torch.zeros_like(increment))))
    assert penalties[0] == pytest.approx(penalties[1], rel=0.02)


def test_biharmonic_is_more_scale_selective(o48: tuple[torch.Tensor, torch.Tensor]) -> None:
    lat, lon = o48
    coarse = _field(torch.cos(lat) ** 2 * torch.cos(2 * lon))  # l = 2
    fine = _field(torch.cos(lat) ** 12 * torch.cos(12 * lon))  # l = 12

    def fine_to_coarse(order: int) -> float:
        loss = _loss(order=order)
        zero = torch.zeros_like(fine)
        return float(loss(fine, fine, reference=zero)) / float(loss(coarse, coarse, reference=zero))

    # Eigenvalue ratio (12*13 / (2*3)) = 26: penalties scale as 26^2 (order 1) and 26^4 (order 2),
    # times the fields' mean-square ratio, which cancels in the order-2 / order-1 comparison.
    assert fine_to_coarse(2) / fine_to_coarse(1) == pytest.approx(26.0**2, rel=1e-3)


def test_min_wavenumber_high_pass(o48: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Wavenumbers below min_wavenumber are free; those at or above it are unchanged."""
    lat, lon = o48
    coarse = _field(torch.cos(lat) ** 2 * torch.cos(2 * lon))  # l = 2
    fine = _field(torch.cos(lat) ** 12 * torch.cos(12 * lon))  # l = 12
    zero = torch.zeros_like(fine)
    full, cut = _loss(), _loss(min_wavenumber=10)
    assert float(cut(coarse, coarse, reference=zero)) == pytest.approx(0.0, abs=1e-12)
    assert float(cut(fine, fine, reference=zero)) == pytest.approx(float(full(fine, fine, reference=zero)), rel=1e-5)
    with pytest.raises(ValueError, match="min_wavenumber"):
        _loss(min_wavenumber=-1)
    with pytest.raises(ValueError, match="min_wavenumber"):
        _loss(min_wavenumber=100_000)


# ── reference contract ────────────────────────────────────────────────────


def test_missing_reference_raises() -> None:
    pred = torch.randn(1, 1, 1, N48, 1)
    with pytest.raises(RuntimeError, match="reference"):
        _loss()(pred, pred)


def test_none_reference_contributes_zero_with_grad() -> None:
    pred = torch.randn(1, 1, 1, N48, 1, requires_grad=True)
    out = _loss()(pred, pred, reference=None)
    assert out == 0.0
    out.backward()  # still in the graph, so CombinedLoss sums stay differentiable


def test_gradient_flows_to_pred_not_reference() -> None:
    pred = torch.randn(1, 1, 1, N48, 1, requires_grad=True)
    reference = torch.randn(1, 1, 1, N48, 1)
    _loss()(pred, pred, reference=reference).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_runs_under_bf16_autocast() -> None:
    """bf16 inputs under autocast give a finite float32 loss and gradient.

    Smoke test only: CPU autocast does not downcast einsum, so the CUDA failure (SHT einsum autocast to bf16,
    then view_as_complex rejecting it) is not reproduced here.
    """
    pred = torch.randn(1, 1, 1, N48, 1, dtype=torch.bfloat16, requires_grad=True)
    reference = torch.randn(1, 1, 1, N48, 1, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = _loss()(pred, pred, reference=reference)
    assert out.dtype == torch.float32
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(pred.grad).all()


def test_non_finite_reference_raises() -> None:
    pred = torch.randn(1, 1, 1, N48, 2)
    reference = pred.clone()
    reference[..., 1] = float("nan")  # e.g. a diagnostic selected by mistake
    with pytest.raises(ValueError, match="prognostic"):
        _loss()(pred, pred, reference=reference)


def test_shape_mismatch_raises() -> None:
    pred = torch.randn(1, 1, 1, N48, 2)
    with pytest.raises(ValueError, match="shape"):
        _loss()(pred, pred, reference=pred[..., :1])


def test_rejects_non_spherical_transform_and_bad_args() -> None:
    assert SpectralIncrementSmoothnessLoss.needs_reference is True
    with pytest.raises(ValueError, match="spherical-harmonic"):
        SpectralIncrementSmoothnessLoss(transform="fft2d")
    with pytest.raises(ValueError, match="order"):
        _loss(order=0)


# ── plumbing: factory, variable mapper, CombinedLoss, method ──────────────


@pytest.fixture
def data_indices() -> IndexCollection:
    config = DictConfig({"forcing": ["f"], "diagnostic": ["d"]})
    return IndexCollection(config, {"a": 0, "b": 1, "f": 2, "d": 3})


def test_variable_mapper_slices_reference(data_indices: IndexCollection) -> None:
    mapped = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.SpectralIncrementSmoothnessLoss",
                "transform": "octahedral_sht",
                "nlat": 96,
                "predicted_variables": ["b"],
            },
        ),
        data_indices=data_indices,
    )
    out_idx = data_indices.model.output.name_to_index
    pred = torch.randn(1, 1, 1, N48, len(out_idx))
    reference = torch.randn_like(pred)
    reference[..., out_idx["d"]] = float("nan")  # unselected diagnostic: must not reach the loss
    target = torch.zeros(1, 1, 1, N48, 4)
    got = mapped(
        pred,
        target,
        reference=reference,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )
    b = out_idx["b"]
    want = _loss()(pred[..., b : b + 1], target[..., :1], reference=reference[..., b : b + 1])
    torch.testing.assert_close(got, want)


def test_combined_loss_sub_loss_does_not_inherit_grid_scalers(data_indices: IndexCollection) -> None:
    """``scalers: []`` keeps the CombinedLoss's node_weights (a grid scaler) off the spectral term."""
    combined = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "scalers": ["node_weights"],
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {
                        "_target_": "anemoi.training.losses.SpectralIncrementSmoothnessLoss",
                        "nlat": 96,
                        "predicted_variables": ["a", "b"],
                        "scalers": [],
                    },
                ],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"node_weights": (3, torch.ones(N48))},
        data_indices=data_indices,
    )
    smooth = next(leaf for leaf in combined.iter_leaf_losses() if getattr(leaf, "needs_reference", False))
    assert "node_weights" not in smooth.scaler.tensors
    pred = torch.randn(1, 1, 1, N48, 3)
    out = combined(
        pred,
        torch.randn(1, 1, 1, N48, 4),
        reference=torch.randn_like(pred),
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )
    assert torch.isfinite(out)


def test_combined_loss_passes_reference() -> None:
    smooth = _loss()
    combined = CombinedLoss(losses=[MSELoss(), smooth], loss_weights=[1.0, 2.0])
    pred = torch.randn(1, 1, 1, N48, 1)
    target = torch.randn_like(pred)
    reference = torch.randn_like(pred)
    got = combined(pred, target, reference=reference)
    want = MSELoss()(pred, target) + 2.0 * smooth(pred, target, reference=reference)
    torch.testing.assert_close(got, want)


def test_compute_loss_metrics_routes_reference_per_dataset() -> None:
    seen = {}

    def fake_dataset_loss(y_pred: torch.Tensor, y: torch.Tensor, dataset_name: str, **kwargs) -> tuple:
        del y
        seen[dataset_name] = kwargs.get("reference", "absent")
        return torch.tensor(1.0), {}, y_pred

    stub = SimpleNamespace(target_dataset_names=["a", "b"], compute_dataset_loss_metrics=fake_dataset_loss)
    ref_a = torch.ones(1)
    preds = {"a": torch.zeros(1), "b": torch.zeros(1)}
    BaseTrainingModule.compute_loss_metrics(stub, preds, preds, reference={"a": ref_a, "b": None})
    assert seen["a"] is ref_a
    assert seen["b"] is None
    seen.clear()
    BaseTrainingModule.compute_loss_metrics(stub, preds, preds)
    assert seen == {"a": "absent", "b": "absent"}


def test_mixin_loss_reference_layout(data_indices: IndexCollection) -> None:
    stub = SimpleNamespace(data_indices={"data": data_indices})
    in_idx = data_indices.model.input.name_to_index
    out_idx = data_indices.model.output.name_to_index
    base = torch.randn(2, 3, 1, 5, len(in_idx))  # 3 input steps
    pred = torch.randn(2, 2, 1, 5, len(out_idx))  # 2 output steps
    ref = DATrainingMixin._loss_reference(stub, {"data": base}, {"data": pred}, cold_start=False)["data"]
    assert ref.shape == pred.shape
    for name in ("a", "b"):
        for t in range(2):
            torch.testing.assert_close(ref[:, t, ..., out_idx[name]], base[:, -1, ..., in_idx[name]])
    assert torch.isnan(ref[..., out_idx["d"]]).all()
    assert DATrainingMixin._loss_reference(stub, {"data": base}, {"data": pred}, cold_start=True) == {"data": None}
