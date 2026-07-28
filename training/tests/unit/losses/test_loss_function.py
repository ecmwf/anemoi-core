# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from types import SimpleNamespace

import einops
import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from pytest_mock import MockerFixture

from anemoi.training.losses import CRPS
from anemoi.training.losses import FourierCorrelationLoss
from anemoi.training.losses import HuberLoss
from anemoi.training.losses import LogCoshLoss
from anemoi.training.losses import LogSpectralDistance
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import RMSELoss
from anemoi.training.losses import SpectralAMSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses import SpectralL2Loss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.enums import TensorDim

spectral_loss_kwargs: dict[type[BaseLoss], dict[str, object]] = {
    SpectralL2Loss: {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
    LogSpectralDistance: {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
    FourierCorrelationLoss: {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
    SpectralCRPSLoss: {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
    SpectralAMSELoss: {"transform": "octahedral_sht", "nlat": 8},
}
spectral_losses = list(spectral_loss_kwargs)
losses = [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, CRPS, WeightedMSELoss, *spectral_losses]


def _resolve_subgrid(cfg: dict, output_mask: SimpleNamespace | None = None) -> None:
    mock_method = SimpleNamespace(output_mask={"data": output_mask})
    multi_cfg = {"data": cfg}
    BaseTrainingModule._resolve_subgrid(mock_method, multi_cfg)
    return multi_cfg["data"]


def _make_loss(target: str, output_mask: SimpleNamespace | None = None, **kwargs) -> BaseLoss:
    cfg = {"_target_": target, "scalers": []}
    cfg.update(kwargs)
    cfg = _resolve_subgrid(cfg, output_mask)
    return get_loss_function(DictConfig(cfg))


def _assert_variable_and_scalar_shapes(
    loss: BaseLoss,
    pred: torch.Tensor,
    target: torch.Tensor,
    nvars: int,
) -> None:
    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,), "squash=False should return per-variable loss"
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1, "squash=True should return a single aggregated loss"


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_manual_init(loss_cls: type[BaseLoss]) -> None:
    loss = loss_cls(**spectral_loss_kwargs.get(loss_cls, {}))
    assert isinstance(loss, BaseLoss)


def _expected_crps(preds: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
    ens_size = preds.shape[-1]
    mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)
    pair_sum = torch.zeros_like(mae)
    for i in range(ens_size - 1):
        pair_sum += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)
    coef = -(alpha / (ens_size * (ens_size - 1)) + (1.0 - alpha) / (ens_size**2))
    return mae + coef * pair_sum


def test_crps_defaults_to_almost_fair_stable_backend() -> None:
    loss = CRPS()
    assert loss.alpha == 0.95
    assert loss.backend == "stable"
    assert loss.name == "crps0.95"


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.95, 1.0])
def test_crps_backends_match_expected_formula(alpha: float) -> None:
    preds = torch.randn(2, 2, 3, 4, 5, dtype=torch.float64)
    targets = torch.randn(2, 2, 3, 4, dtype=torch.float64)

    expected = _expected_crps(preds, targets, alpha)
    naive = CRPS(alpha=alpha, backend="naive")._kernel_crps(preds, targets)
    stable = CRPS(alpha=alpha, backend="stable")._kernel_crps(preds, targets)

    torch.testing.assert_close(naive, expected)
    torch.testing.assert_close(stable, expected)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_crps_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in the range"):
        CRPS(alpha=alpha)


def test_crps_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match="Unknown CRPS backend"):
        CRPS(backend="unknown")  # type: ignore[arg-type]


def test_crps_with_singleton_target_ensemble_dim() -> None:
    pred = torch.zeros(2, 1, 4, 3, 2)
    target = torch.zeros(2, 1, 1, 3, 2)

    out = CRPS()(pred, target, squash=False)

    torch.testing.assert_close(out, torch.zeros(2))


def test_crps_mask_nans_preserves_ensemble_sizes() -> None:
    pred = torch.ones(1, 1, 3, 2, 1)
    target = torch.ones(1, 1, 2, 2, 1)
    pred[:, :, 1, 0, 0] = torch.nan
    target[:, :, 0, 1, 0] = torch.nan

    masked_pred, masked_target = CRPS(ignore_nans=True).mask_nans(pred, target)

    assert masked_pred.shape == pred.shape
    assert masked_target.shape == target.shape
    assert torch.isfinite(masked_pred).all()
    assert torch.isfinite(masked_target).all()


@pytest.fixture
def functionalloss() -> type[FunctionalLoss]:
    class ReturnDifference(FunctionalLoss):
        def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return pred - target

    return ReturnDifference


@pytest.fixture
def loss_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 1, 1, 4, 2]

    pred = torch.zeros(tensor_shape)
    pred[0, 0, 0, 0] = torch.tensor([1.0, 1.0])
    target = torch.zeros(tensor_shape)

    # With only one "grid point" differing by 1 in all
    # variables, the loss should be 1.0

    loss_result = torch.tensor([1.0])
    return pred, target, loss_result


@pytest.fixture
def loss_inputs_fine(
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs with finer grid."""
    pred, target, loss_result = loss_inputs

    pred = torch.cat([pred, pred], dim=2)
    target = torch.cat([target, target], dim=2)

    return pred, target, loss_result


def test_assert_of_grid_dim(functionalloss: type[FunctionalLoss]) -> None:
    """Test that the grid dimension is set correctly."""
    loss = functionalloss()
    loss.add_scaler(TensorDim.VARIABLE, 1.0, name="variable_test")

    assert TensorDim.GRID not in loss.scaler, "Grid dimension should not be set"

    with pytest.raises(RuntimeError):
        loss.scale(torch.ones((4, 2)))


@pytest.mark.parametrize("add_grid_scaler", [False, True])
def test_scale_subset_indices_requires_tuple(
    functionalloss: type[FunctionalLoss],
    add_grid_scaler: bool,
) -> None:
    loss = functionalloss()
    if add_grid_scaler:
        loss.add_scaler(TensorDim.GRID, torch.tensor([1.0, 2.0, 3.0, 4.0]), name="grid_test")

    x = torch.arange(1 * 1 * 1 * 4 * 5, dtype=torch.float32).reshape(1, 1, 1, 4, 5)
    with pytest.raises(TypeError, match="must be a tuple"):
        loss.scale(x, subset_indices=[Ellipsis, [1, 3]])


@pytest.fixture
def simple_functionalloss(functionalloss: type[FunctionalLoss]) -> FunctionalLoss:
    loss = functionalloss()
    loss.add_scaler(TensorDim.GRID, torch.ones((4,)), name="unit_scaler")
    return loss


@pytest.fixture
def functionalloss_with_scaler(simple_functionalloss: FunctionalLoss) -> FunctionalLoss:
    loss = simple_functionalloss
    loss.add_scaler(TensorDim.GRID, torch.rand((4,)), name="test")
    return loss


@pytest.fixture
def functionalloss_with_scaler_fine(functionalloss: FunctionalLoss) -> FunctionalLoss:
    loss = functionalloss()
    loss.add_scaler(TensorDim.GRID, torch.rand((8,)), name="test")
    return loss


def test_simple_functionalloss(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test a functional loss."""
    pred, target, loss_result = loss_inputs

    loss = simple_functionalloss(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, loss_result), "Loss should be equal to the expected result"


def test_batch_invariance(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, loss_result = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 4

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape, "Batch size should be different"

    loss_batch_size_1 = simple_functionalloss(pred_batch_size_1, target_batch_size_1)
    loss_batch_size_2 = simple_functionalloss(pred_batch_size_2, target_batch_size_2)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert torch.allclose(loss_batch_size_1, loss_result), "Loss should be equal to the expected result"

    assert isinstance(loss_batch_size_2, torch.Tensor)
    assert torch.allclose(loss_batch_size_2, loss_result), "Loss should be equal to the expected result"

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_batch_invariance_without_squash(
    simple_functionalloss: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, _ = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape, "Batch size should be different"

    loss_batch_size_1 = simple_functionalloss(pred_batch_size_1, target_batch_size_1, squash=False)
    loss_batch_size_2 = simple_functionalloss(pred_batch_size_2, target_batch_size_2, squash=False)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert isinstance(loss_batch_size_2, torch.Tensor)

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_batch_invariance_with_scaler(
    functionalloss_with_scaler: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    pred, target, _ = loss_inputs

    pred_batch_size_1 = pred
    target_batch_size_1 = target

    new_shape = list(pred.shape)
    new_shape[0] = 2

    pred_batch_size_2 = pred.expand(new_shape)
    target_batch_size_2 = target.expand(new_shape)

    assert pred_batch_size_1.shape != pred_batch_size_2.shape

    loss_batch_size_1 = functionalloss_with_scaler(pred_batch_size_1, target_batch_size_1)
    loss_batch_size_2 = functionalloss_with_scaler(pred_batch_size_2, target_batch_size_2)

    assert isinstance(loss_batch_size_1, torch.Tensor)
    assert isinstance(loss_batch_size_2, torch.Tensor)

    assert torch.allclose(loss_batch_size_1, loss_batch_size_2), "Losses should be equal between batch sizes"


def test_grid_invariance(
    functionalloss_with_scaler: FunctionalLoss,
    functionalloss_with_scaler_fine: FunctionalLoss,
    loss_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test for batch invariance."""
    gdim = TensorDim.GRID
    pred_coarse, target_coarse, _ = loss_inputs
    pred_fine = torch.cat([pred_coarse, pred_coarse], dim=gdim)
    target_fine = torch.cat([target_coarse, target_coarse], dim=gdim)

    num_points_coarse = pred_coarse.shape[gdim]
    num_points_fine = pred_fine.shape[gdim]

    functionalloss_with_scaler.update_scaler("test", torch.ones((num_points_coarse,)) / num_points_coarse)
    functionalloss_with_scaler_fine.update_scaler("test", torch.ones((num_points_fine,)) / num_points_fine)

    loss_coarse = functionalloss_with_scaler(pred_coarse, target_coarse)
    loss_fine = functionalloss_with_scaler_fine(pred_fine, target_fine)

    assert isinstance(loss_coarse, torch.Tensor)
    assert isinstance(loss_fine, torch.Tensor)

    assert torch.allclose(loss_coarse, loss_fine), "Losses should be equal between grid sizes"


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_include(loss_cls: type[BaseLoss]) -> None:
    loss_dic = {
        "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        **spectral_loss_kwargs.get(loss_cls, {}),
    }
    loss = get_loss_function(DictConfig(loss_dic))
    assert isinstance(loss, BaseLoss)


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler(loss_cls: type[BaseLoss]) -> None:
    loss_dic = {
        "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        **spectral_loss_kwargs.get(loss_cls, {}),
        "scalers": ["test"],
    }
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_add_all(loss_cls: type[BaseLoss]) -> None:
    loss_dic = {
        "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        **spectral_loss_kwargs.get(loss_cls, {}),
        "scalers": ["*"],
    }
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": ((0, 1), torch.ones((1, 2)))},
    )
    assert isinstance(loss, BaseLoss)

    assert "test" in loss.scaler
    torch.testing.assert_close(loss.scaler.get_scaler(2), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler_not_add(loss_cls: type[BaseLoss]) -> None:
    loss_dic = {
        "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        **spectral_loss_kwargs.get(loss_cls, {}),
        "scalers": [],
    }
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler_exclude(loss_cls: type[BaseLoss]) -> None:
    loss_dic = {
        "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        **spectral_loss_kwargs.get(loss_cls, {}),
        "scalers": ["*", "!test"],
    }
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


@pytest.mark.parametrize(
    "target",
    [
        "anemoi.training.losses.spectral.LogFFT2Distance",
        "anemoi.training.losses.spectral.FourierCorrelationLoss",
    ],
)
def test_fft2d_spectral_losses_shape_and_validation(target: str) -> None:
    """FFT2D spectral losses should produce expected output shapes and validate grid size."""
    loss = _make_loss(target, x_dim=710, y_dim=640)
    assert isinstance(loss, BaseLoss)
    assert hasattr(loss.transform, "x_dim")
    assert hasattr(loss.transform, "y_dim")

    # pred/target are (batch, steps, grid, vars)
    pred = torch.ones((6, 1, 1, 710 * 640, 2))
    target_tensor = torch.zeros((6, 1, 1, 710 * 640, 2))
    _assert_variable_and_scalar_shapes(loss, pred, target_tensor, nvars=2)

    # wrong grid size should fail (FFT2D reshape/assert)
    wrong = (torch.ones((6, 1, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 1, 710 * 640 + 1, 2)))
    with pytest.raises(einops.EinopsError):
        _ = loss(*wrong, squash=True)


@pytest.mark.parametrize(
    ("loss_cls", "ensemble_size", "transform_kwargs", "grid_shard_sizes", "spectral_shard_sizes"),
    [
        pytest.param(
            SpectralL2Loss,
            1,
            {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
            [8, 8],
            [8, 8],
            id="fft2d-l2",
        ),
        pytest.param(
            LogSpectralDistance,
            1,
            {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
            [8, 8],
            [8, 8],
            id="fft2d-log",
        ),
        pytest.param(
            FourierCorrelationLoss,
            1,
            {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
            [8, 8],
            [8, 8],
            id="fft2d-fcl",
        ),
        pytest.param(
            SpectralCRPSLoss,
            4,
            {"transform": "fft2d", "x_dim": 4, "y_dim": 4},
            [8, 8],
            [8, 8],
            id="fft2d-crps",
        ),
        pytest.param(
            SpectralAMSELoss,
            1,
            {"transform": "octahedral_sht", "nlat": 8},
            [104, 104],
            [2, 2],
            id="octahedral-sht-amse",
        ),
    ],
)
def test_spectral_losses_use_all_to_all_for_sharded_layout(
    loss_cls: type[BaseLoss],
    ensemble_size: int,
    transform_kwargs: dict[str, object],
    grid_shard_sizes: list[int],
    spectral_shard_sizes: list[int],
    mocker: MockerFixture,
) -> None:
    group = object()
    channel_shard_sizes = [1, 1]
    local_grid = grid_shard_sizes[0]

    loss = loss_cls(**transform_kwargs)
    pred = torch.randn(1, 1, ensemble_size, local_grid, 2)
    target = torch.randn(1, 1, 1, local_grid, 2)

    def fake_all_to_all(
        x: torch.Tensor,
        dim_split: int,
        split_sizes: list[int],
        dim_concat: int,
        concat_sizes: list[int],
        _group: object,
    ) -> torch.Tensor:
        out_shape = list(x.shape)
        dim_split %= x.ndim
        dim_concat %= x.ndim
        out_shape[dim_split] = split_sizes[0]
        out_shape[dim_concat] = sum(concat_sizes)
        return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

    get_sizes = mocker.patch(
        "anemoi.training.losses.spectral.get_shard_sizes",
        side_effect=[channel_shard_sizes, channel_shard_sizes, spectral_shard_sizes],
    )
    all_to_all = mocker.patch(
        "anemoi.training.losses.spectral.all_to_all_transpose",
        side_effect=fake_all_to_all,
    )
    mocker.patch("anemoi.training.losses.base.reduce_tensor", side_effect=lambda x, _group: x)
    scale = mocker.spy(loss, "scale")

    out = loss(
        pred,
        target,
        group=group,
        grid_shard_slice=slice(0, local_grid),
        grid_shard_sizes=grid_shard_sizes,
        grid_dim=-2,
    )

    assert torch.isfinite(out).all()
    assert get_sizes.call_count == 3
    assert get_sizes.call_args_list[0].args[1] == TensorDim.VARIABLE
    assert get_sizes.call_args_list[1].args[1] == TensorDim.VARIABLE
    assert get_sizes.call_args_list[2].args[1] == -2

    assert all_to_all.call_count == 3
    assert all_to_all.call_args_list[0].args[1:] == (
        TensorDim.VARIABLE,
        channel_shard_sizes,
        -2,
        grid_shard_sizes,
        group,
    )
    assert all_to_all.call_args_list[1].args[1:] == (
        TensorDim.VARIABLE,
        channel_shard_sizes,
        -2,
        grid_shard_sizes,
        group,
    )
    assert all_to_all.call_args_list[2].args[1:] == (
        -2,
        spectral_shard_sizes,
        TensorDim.VARIABLE,
        channel_shard_sizes,
        group,
    )
    assert scale.call_args.kwargs["grid_shard_slice"] is None


@pytest.mark.parametrize("loss_cls", spectral_losses)
def test_spectral_losses_report_sharding_support(loss_cls: type[BaseLoss]) -> None:
    loss = loss_cls(**spectral_loss_kwargs[loss_cls])
    assert loss.supports_sharding is True
    assert loss.needs_shard_layout_info is True


def test_iter_leaf_losses_flat() -> None:
    """Test that iter_leaf_losses on a simple loss yields itself."""
    loss = MSELoss()
    leaves = list(loss.iter_leaf_losses())
    assert len(leaves) == 1
    assert leaves[0] is loss


def _octahedral_expected_points(nlat: int) -> int:
    half = [4 * (i + 1) + 16 for i in range(nlat // 2)]
    nlon = half + half[::-1]
    return int(sum(nlon))


def test_spectral_amse_octahedral_sht_shapes_and_transform_validation() -> None:
    nlat = 8
    nvars = 3
    expected_points = _octahedral_expected_points(nlat)

    loss = _make_loss("anemoi.training.losses.spectral.SpectralAMSELoss", transform="octahedral_sht", nlat=nlat)
    pred = torch.zeros((2, 1, 1, expected_points, nvars))
    target = torch.zeros_like(pred)
    _assert_variable_and_scalar_shapes(loss, pred, target, nvars=nvars)


@pytest.mark.parametrize("transform", ["fft2d", "dct2d"])
def test_amse_cartesian_transforms(transform: str) -> None:
    """AMSE works with the 2D transforms via radial-wavenumber binning."""
    nvars = 3
    x_dim, y_dim = 8, 6
    points = x_dim * y_dim

    loss = _make_loss(
        "anemoi.training.losses.spectral.SpectralAMSELoss",
        transform=transform,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    # shapes (squash False -> per-variable, squash True -> scalar)
    pred = torch.zeros((2, 1, 1, points, nvars))
    target = torch.zeros_like(pred)
    _assert_variable_and_scalar_shapes(loss, pred, target, nvars=nvars)

    # AMSE(x, x) == 0 and AMSE >= 0 (fp64 to avoid the loss's known fp32 sqrt/square
    # eps-rounding, which is transform-agnostic)
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 1, points, nvars, dtype=torch.float64)
    other = torch.randn(2, 1, 1, points, nvars, dtype=torch.float64)
    assert abs(loss(pred, pred, squash=True).item()) < 1e-6
    assert loss(pred, pred + other, squash=True).item() >= -1e-9

    # the transform's spectral-density contract (Parseval partition, cross self-consistency)
    # is covered for all transforms in models/tests/layers/test_spectral_density.py

    # gradients flow back through the radial-binning path (index_add_); the existing
    # backward tests cover SpectralCRPS / SpectralL2Loss, not AMSE's binning
    pred = pred.clone().requires_grad_(True)
    loss(pred, target.double(), squash=True).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # patch-wise FFT2D has no single (ky, kx) plane to bin, so AMSE must reject it
    if transform == "fft2d":
        with pytest.raises(AssertionError, match="does not support patch-wise FFT2D"):
            SpectralAMSELoss(
                transform="fft2d",
                x_dim=x_dim,
                y_dim=y_dim,
                patch_size=[3, 4],
            )


def test_spectral_l2_octahedral_sht_shapes_and_grid_validation() -> None:

    nlat = 8
    nvars = 3
    expected_points = _octahedral_expected_points(nlat)

    loss = _make_loss("anemoi.training.losses.spectral.SpectralL2Loss", transform="octahedral_sht", nlat=nlat)
    pred = torch.zeros((2, 1, 1, expected_points, nvars))
    target = torch.zeros_like(pred)
    _assert_variable_and_scalar_shapes(loss, pred, target, nvars=nvars)
    pred_wrong = torch.zeros((2, 1, 1, expected_points + 1, nvars))
    target_wrong = torch.zeros_like(pred_wrong)
    with pytest.raises(AssertionError):
        _ = loss(pred_wrong, target_wrong, squash=True)


@pytest.mark.parametrize("transform", ["fft2d", "dct2d"])
def test_spectral_crps_cartesian_transform(transform: str) -> None:
    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    loss = _make_loss(
        "anemoi.training.losses.spectral.SpectralCRPSLoss",
        transform=transform,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    _assert_variable_and_scalar_shapes(loss, pred, target, nvars=nvars)


def test_spectral_crps_fft2d_projection(mocker: MockerFixture) -> None:
    from scipy.sparse import eye

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    sparse_mat = eye(grid, format="csr")
    mocker.patch("anemoi.models.layers.graph_provider.load_npz", return_value=sparse_mat)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {"matrix_path": "/path/to/projection_matrix.npz"},
                "scalers": [],
            },
        ),
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,), "fft2d: per-variable CRPS expected"
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1, "fft2d: scalar CRPS expected"


def test_spectral_loss_projection_actually_applied(mocker: MockerFixture) -> None:
    """Projection must be applied: a non-square matrix (n_src→n_dst) is used, FFT2D.

    FFT2D is configured for n_dst. If projection is skipped the reshape raises EinopsError.
    """
    from scipy.sparse import csr_matrix

    n_src, x_dim, y_dim = 12, 4, 2  # 12 input nodes, project down to 8
    n_dst = x_dim * y_dim
    bs, nvars = 1, 2

    # Simple non-square projection: first n_dst rows of identity (drop last 4 nodes)
    proj = csr_matrix(np.eye(n_dst, n_src, dtype=np.float32))
    mocker.patch("anemoi.models.layers.graph_provider.load_npz", return_value=proj)

    loss = SpectralL2Loss(
        transform="fft2d",
        x_dim=x_dim,
        y_dim=y_dim,
        projection_config={"matrix_path": "/fake/path.npz"},
    )

    pred = torch.randn(bs, 1, 1, n_src, nvars)
    target = torch.randn(bs, 1, 1, n_src, nvars)
    result = loss(pred, target)
    assert result.numel() == 1


@pytest.mark.parametrize(
    "subgrid",
    [
        (0, 8),
        "output_mask",
    ],
)
def test_spectral_loss_subgrid_actually_applied(subgrid: str | tuple) -> None:
    """Subgrid must be applied: input has 2x the expected nodes, slice selects half.

    If subgrid is skipped FFT2D fails to reshape the oversized spatial dimension.
    """
    x_dim, y_dim = 4, 2  # FFT2D expects 8 nodes
    n_total = 16  # input has 16 nodes; slice=(0, 8) should reduce to 8
    bs, nvars = 1, 2
    loss_cfg = {
        "transform": "fft2d",
        "x_dim": x_dim,
        "y_dim": y_dim,
        "subgrid": subgrid,
    }

    output_mask = SimpleNamespace(as_tuple=lambda: (0, 8))

    loss = _make_loss("anemoi.training.losses.spectral.SpectralL2Loss", output_mask=output_mask, **loss_cfg)

    pred = torch.randn(bs, 1, 1, n_total, nvars)
    target = torch.randn(bs, 1, 1, n_total, nvars)
    result = loss(pred, target)
    assert result.numel() == 1


def test_spectral_loss_projection_wrong_output_size_raises(mocker: MockerFixture) -> None:
    """Projection that outputs wrong node count should raise on FFT2D reshape."""
    from scipy.sparse import csr_matrix

    n_src, x_dim, y_dim = 12, 4, 2  # FFT2D expects 8 nodes
    n_wrong = 10  # projection outputs 10 nodes, not 8
    proj = csr_matrix(np.eye(n_wrong, n_src, dtype=np.float32))
    mocker.patch("anemoi.models.layers.graph_provider.load_npz", return_value=proj)

    loss = SpectralL2Loss(
        transform="fft2d",
        x_dim=x_dim,
        y_dim=y_dim,
        projection_config={"matrix_path": "/fake/path.npz"},
    )
    pred = torch.randn(1, 1, 1, n_src, 2)
    target = torch.randn(1, 1, 1, n_src, 2)
    with pytest.raises(einops.EinopsError):
        loss(pred, target)


def test_spectral_loss_subgrid_out_of_bounds_raises() -> None:
    """Subgrid that requests more nodes than available should raise."""
    x_dim, y_dim = 4, 2  # expects 8 nodes
    n_total = 6  # fewer nodes than slice end requests

    loss = SpectralL2Loss(
        transform="fft2d",
        x_dim=x_dim,
        y_dim=y_dim,
        subgrid=(0, 8),  # requests 8 nodes but only 6 exist
    )
    pred = torch.randn(1, 1, 1, n_total, 2)
    target = torch.randn(1, 1, 1, n_total, 2)

    with pytest.raises(einops.EinopsError):
        loss(pred, target)


def test_spectral_loss_ambiguous_projection_config_raises() -> None:
    """Specifying both matrix_path and edges_name in projection_config should raise."""
    with pytest.raises(ValueError, match="at most one of"):
        SpectralL2Loss(
            transform="fft2d",
            x_dim=4,
            y_dim=2,
            projection_config={
                "matrix_path": "/fake/path.npz",
                "edges_name": ("data", "to", "target"),
            },
        )


def test_spectral_crps_projection_applies_subgrid_before_projection(mocker: MockerFixture) -> None:
    from scipy.sparse import eye

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    projected_grid = x_dim * y_dim
    source_grid = projected_grid * 2

    pred = torch.randn(bs, 1, ens, source_grid, nvars)
    target = torch.randn(bs, 1, 1, source_grid, nvars)

    mocker.patch("anemoi.models.layers.graph_provider.load_npz", return_value=eye(projected_grid, format="csr"))

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "subgrid": (0, projected_grid),
                "projection_config": {"matrix_path": "/path/to/projection_matrix.npz"},
                "scalers": [],
            },
        ),
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)


def test_spectral_crps_projection_from_graph_config() -> None:
    from torch_geometric.data import HeteroData

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 2, 2
    grid = x_dim * y_dim

    graph = HeteroData()
    graph["data"].x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.017453292],
            [0.017453292, 0.0],
            [0.017453292, 0.017453292],
        ],
        dtype=torch.float32,
    )
    graph["data"].num_nodes = grid

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.LatLonNodes",
                        "latitudes": [0.0, 0.0, 1.0, 1.0],
                        "longitudes": [0.0, 1.0, 0.0, 1.0],
                    },
                    "num_nearest_neighbours": 3,
                    "sigma": 0.01,
                    "row_normalize": False,
                },
                "scalers": [],
            },
        ),
        graph_data=graph,
        data_node_name="data",
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)

    # Target-grid mode applies the Gaussian (sigma-weighted) KNN weights by default; a
    # uniform fallback (the regression) would make every non-zero edge weight identical.
    weights = loss.projection_provider.get_edges().to_dense()
    assert weights[weights != 0].std() > 1e-6


def test_spectral_crps_projection_from_existing_edges() -> None:
    from torch_geometric.data import HeteroData

    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 2, 2
    grid = x_dim * y_dim
    edges_name = ("data", "to", "projection")

    graph = HeteroData()
    graph["data"].num_nodes = grid
    graph["projection"].num_nodes = grid
    graph[edges_name].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=torch.long,
    )
    graph[edges_name].gauss_weight = torch.ones(grid)

    pred = torch.randn(bs, 1, ens, grid, nvars)
    target = torch.randn(bs, 1, 1, grid, nvars)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                "transform": "fft2d",
                "x_dim": x_dim,
                "y_dim": y_dim,
                "projection_config": {
                    "edges_name": edges_name,
                    "edge_weight_attribute": "gauss_weight",
                },
                "scalers": [],
            },
        ),
        graph_data=graph,
        data_node_name="data",
    )

    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)


@pytest.mark.parametrize(
    ("transform", "transform_kwargs"),
    [
        pytest.param("octahedral_sht", {"nlat": 8}, id="octahedral"),
        pytest.param("reduced_sht", {"grid": "n320", "truncation": 3}, id="reduced"),
    ],
)
def test_spectral_crps_sht_transforms(
    transform: str,
    transform_kwargs: dict[str, object],
    mocker: MockerFixture,
) -> None:
    ring_sizes = [20, 24, 28, 32, 32, 28, 24, 20]
    if transform == "reduced_sht":
        # Use a small reduced grid while exercising the real ReducedSHT transform.
        latitudes = np.repeat(np.arange(len(ring_sizes)), ring_sizes)
        mocker.patch("anemoi.transform.grids.named.lookup", return_value={"latitudes": latitudes})

    nvars = 2
    pred = torch.randn(2, 1, 4, sum(ring_sizes), nvars, requires_grad=True)
    target = torch.randn(2, 1, 1, sum(ring_sizes), nvars)
    loss = _make_loss(
        "anemoi.training.losses.spectral.SpectralCRPSLoss",
        transform=transform,
        **transform_kwargs,
    )

    per_variable = loss(pred, target, squash=False)
    assert per_variable.shape == (nvars,)
    assert torch.isfinite(per_variable).all()

    scalar = loss(pred, target, squash=True)
    assert scalar.numel() == 1
    assert torch.isfinite(scalar).all()
    scalar.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_mse_ignore_nans() -> None:
    """MSELoss should ignore NaNs with ignore_nans=True."""
    pred = torch.randn(3, 4, 5, 6, 7)
    pred.requires_grad_()
    target = torch.randn(3, 4, 5, 6, 7)
    target[..., 0, 0] = torch.nan

    loss = MSELoss(ignore_nans=True)

    out = loss(pred, target)
    assert torch.isfinite(out).all(), "Expected finite loss with ignore_nans=True"

    (grad,) = torch.autograd.grad(out, pred, retain_graph=True)
    assert torch.isfinite(grad).all(), "Expected finite gradients"


def test_mse_nans() -> None:
    """MSELoss should propagate NaNs with ignore_nans=False."""
    pred = torch.randn(3, 4, 5, 6, 7)
    pred.requires_grad_()
    target = torch.randn(3, 4, 5, 6, 7)
    target[..., 0, 0] = torch.nan

    loss = MSELoss(ignore_nans=False)

    out = loss(pred, target)
    assert torch.isnan(out).any(), "Expected nan loss with ignore_nans=False"
