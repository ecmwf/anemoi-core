# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import einops
import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.losses import AlmostFairKernelCRPS
from anemoi.training.losses import FourierCorrelationLoss
from anemoi.training.losses import HuberLoss
from anemoi.training.losses import KernelCRPS
from anemoi.training.losses import LogCoshLoss
from anemoi.training.losses import LogSpectralDistance
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import RMSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses import SpectralL2Loss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.utils.enums import TensorDim

losses = [MSELoss, HuberLoss, MAELoss, RMSELoss, LogCoshLoss, KernelCRPS, AlmostFairKernelCRPS, WeightedMSELoss]
spectral_losses = [SpectralL2Loss, SpectralCRPSLoss, FourierCorrelationLoss, LogSpectralDistance]
losses += spectral_losses


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_manual_init(loss_cls: type[BaseLoss]) -> None:
    loss = loss_cls(x_dim=4, y_dim=4) if loss_cls in spectral_losses else loss_cls()
    assert isinstance(loss, BaseLoss)


@pytest.fixture
def functionalloss() -> type[FunctionalLoss]:
    class ReturnDifference(FunctionalLoss):
        def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return pred - target

    return ReturnDifference


@pytest.fixture
def loss_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture for loss inputs."""
    tensor_shape = [1, 1, 4, 2]

    pred = torch.zeros(tensor_shape)
    pred[0, 0, 0] = torch.tensor([1.0, 1.0])
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
    new_shape[0] = 2

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
    pred_coarse, target_coarse, _ = loss_inputs
    pred_fine = torch.cat([pred_coarse, pred_coarse], dim=2)
    target_fine = torch.cat([target_coarse, target_coarse], dim=2)

    num_points_coarse = pred_coarse.shape[2]
    num_points_fine = pred_fine.shape[2]

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
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "x_dim": 4,
            "y_dim": 4,
        }
    )
    loss = get_loss_function(DictConfig(loss_dic))
    assert isinstance(loss, BaseLoss)


@pytest.mark.parametrize(
    "loss_cls",
    losses,
)
def test_dynamic_init_scaler(loss_cls: type[BaseLoss]) -> None:
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["test"],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["test"],
            "x_dim": 4,
            "y_dim": 4,
        }
    )
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
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["*"],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["*"],
            "x_dim": 4,
            "y_dim": 4,
        }
    )
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
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": [],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": [],
            "x_dim": 4,
            "y_dim": 4,
        }
    )
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
    loss_dic = (
        {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "scalers": ["*", "!test"],
        }
        if loss_cls not in spectral_losses
        else {
            "_target_": f"anemoi.training.losses.{loss_cls.__name__}",
            "x_dim": 4,
            "y_dim": 4,
            "scalers": ["*", "!test"],
        }
    )
    loss = get_loss_function(
        DictConfig(loss_dic),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss, BaseLoss)
    assert "test" not in loss.scaler


def test_logfft2dist_loss() -> None:
    """Test that LogFFT2Distance can be instantiated and validates input shape."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.LogFFT2Distance",
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
    )
    assert isinstance(loss, BaseLoss)
    assert hasattr(loss, "x_dim")
    assert hasattr(loss, "y_dim")

    # pred/target are (batch, steps, grid, vars)
    # TODO (Ophelia): edit this when multi ouptuts get merged
    right = (torch.ones((6, 1, 710 * 640, 2)), torch.zeros((6, 1, 710 * 640, 2)))

    # squash=False -> per-variable loss
    loss_value = loss(*right, squash=False)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.ndim == 1 and loss_value.shape[0] == 2, "Expected per-variable loss (n_vars,)"

    # squash=True -> single aggregated loss
    loss_total = loss(*right, squash=True)
    assert isinstance(loss_total, torch.Tensor)
    assert loss_total.numel() == 1, "Expected a single aggregated loss value"

    # wrong grid size should fail (FFT2D reshape/assert)
    wrong = (torch.ones((6, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 710 * 640 + 1, 2)))
    with pytest.raises(einops.EinopsError):
        _ = loss(*wrong, squash=True)


def test_fcl_loss() -> None:
    # TODO (Ophelia): edit this when multi ouptuts get merged
    """Test that FourierCorrelationLoss can be instantiated and validates input shape."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.FourierCorrelationLoss",
                "x_dim": 710,
                "y_dim": 640,
                "scalers": [],
            },
        ),
    )
    assert isinstance(loss, BaseLoss)
    assert hasattr(loss, "x_dim")
    assert hasattr(loss, "y_dim")

    right = (torch.ones((6, 1, 710 * 640, 2)), torch.zeros((6, 1, 710 * 640, 2)))

    loss_value = loss(*right, squash=False)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.ndim == 1 and loss_value.shape[0] == 2, "Expected per-variable loss (n_vars,)"

    loss_total = loss(*right, squash=True)
    assert isinstance(loss_total, torch.Tensor)
    assert loss_total.numel() == 1, "Expected a single aggregated loss value"

    wrong = (torch.ones((6, 1, 710 * 640 + 1, 2)), torch.zeros((6, 1, 710 * 640 + 1, 2)))
    with pytest.raises(AssertionError):
        _ = loss(*wrong, squash=True)


def test_octahedral_sht_loss() -> None:
    def _octahedral_expected_points(nlat: int) -> int:
        half = [4 * (i + 1) + 16 for i in range(nlat // 2)]
        nlon = half + half[::-1]
        return int(sum(nlon))

    nlat = 8
    nvars = 3
    expected_points = _octahedral_expected_points(nlat)

    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralL2Loss",
                "transform": "octahedral_sht",
                "y_dim": nlat,
                "scalers": [],
            },
        ),
    )
    pred = torch.zeros((2, 1, expected_points, nvars))
    target = torch.zeros_like(pred)
    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,), "squash=False should return per-variable loss"
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1, "squash=True should return a single aggregated loss"
    pred_wrong = torch.zeros((2, 1, expected_points + 1, nvars))
    target_wrong = torch.zeros_like(pred_wrong)
    with pytest.raises(AssertionError):
        _ = loss(pred_wrong, target_wrong, squash=True)


def test_cartesian_sht_loss() -> None:
    nlat = 8
    nlon = 16
    nvars = 3
    expected_points = nlat * nlon
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.spectral.SpectralL2Loss",
                "transform": "cartesian_sht",
                "x_dim": nlon,
                "y_dim": nlat,
                "grid": "legendre-gauss",
                "scalers": [],
            },
        ),
    )

    pred = torch.zeros((2, 1, expected_points, nvars))
    target = torch.zeros_like(pred)
    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1
    pred_wrong = torch.zeros((2, 1, expected_points + 1, nvars))
    target_wrong = torch.zeros_like(pred_wrong)
    with pytest.raises(AssertionError):
        _ = loss(pred_wrong, target_wrong, squash=True)


def _expected_octahedral_points(truncation: int) -> int:
    # full globe reduced-octahedral points for ecTrans definition
    # NH lons: 20 + 4*i, i=0..T  => sum_NH = 2*(T+1)*(T+10)
    # full globe doubles:        => 4*(T+1)*(T+10)
    return 4 * (truncation + 1) * (truncation + 10)


class _DummyEcTransOctahedralSHTModule(torch.nn.Module):
    """Stub that avoids needing ectrans assets/npz but preserves shapes."""

    def __init__(self, truncation: int, dtype: torch.dtype = torch.float32, filepath: str | None = None) -> None:
        super().__init__()
        self.truncation = int(truncation)
        self.dtype = dtype
        self.n_grid_points = _expected_octahedral_points(self.truncation)
        self.filepath = filepath

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ens, _, nvars = x.shape
        l_dim = self.truncation + 1
        m_dim = self.truncation + 1

        complex_dtype = torch.complex64 if x.dtype in (torch.float16, torch.float32) else torch.complex128
        return torch.zeros((bsz, ens, l_dim, m_dim, nvars), device=x.device, dtype=complex_dtype)


def test_spectral_l2_loss_ectrans_octahedral_sht_no_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    import anemoi.models.layers.spectral_transforms as st
    from anemoi.training.losses.spectral import SpectralL2Loss

    monkeypatch.setattr(st, "EcTransOctahedralSHTModule", _DummyEcTransOctahedralSHTModule)

    trunc = 8
    nvars = 3
    points = _expected_octahedral_points(trunc)

    # Provide x_dim/y_dim “compat args” (validated by the transform)
    x_dim = 20 + 4 * trunc  # max nlon (equator)
    y_dim = 2 * (trunc + 1)  # number of latitude rings (full globe)

    loss = SpectralL2Loss(
        transform="ectrans_octahedral_sht",
        truncation=trunc,
        x_dim=x_dim,
        y_dim=y_dim,
        # filepath=None is fine because we stub the module
    )

    pred = torch.zeros((2, 1, points, nvars), dtype=torch.float32)
    target = torch.zeros_like(pred)
    out = loss(pred, target, squash=False)
    assert out.shape == (nvars,)
    torch.testing.assert_close(out, torch.zeros((nvars,)))
    out_total = loss(pred, target, squash=True)
    assert out_total.numel() == 1
    torch.testing.assert_close(out_total, torch.tensor(0.0))
    pred_wrong = torch.zeros((2, 1, points + 1, nvars), dtype=torch.float32)
    target_wrong = torch.zeros_like(pred_wrong)
    with pytest.raises(AssertionError):
        _ = loss(pred_wrong, target_wrong, squash=True)


def test_ectrans_octahedral_sht_dim_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the x_dim/y_dim alias checks (even though the grid isn't rectangular)."""
    import anemoi.models.layers.spectral_transforms as st
    from anemoi.training.losses.spectral import SpectralL2Loss

    monkeypatch.setattr(st, "EcTransOctahedralSHTModule", _DummyEcTransOctahedralSHTModule)

    trunc = 8
    x_dim_ok = 20 + 4 * trunc
    y_dim_ok = 2 * (trunc + 1)
    _ = SpectralL2Loss(transform="ectrans_octahedral_sht", truncation=trunc, x_dim=x_dim_ok, y_dim=y_dim_ok)
    with pytest.raises(ValueError, match=r"y_dim.* incompatible with truncation"):
        _ = SpectralL2Loss(transform="ectrans_octahedral_sht", truncation=trunc, x_dim=x_dim_ok, y_dim=y_dim_ok + 1)
    with pytest.raises(ValueError, match=r"x_dim.* incompatible with truncation"):
        _ = SpectralL2Loss(transform="ectrans_octahedral_sht", truncation=trunc, x_dim=x_dim_ok + 1, y_dim=y_dim_ok)


def test_spectral_crps_fft_and_dct() -> None:
    bs, ens, nvars = 2, 5, 3
    x_dim, y_dim = 8, 6
    grid = x_dim * y_dim

    pred = torch.randn(bs, ens, grid, nvars)
    target = torch.randn(bs, 1, grid, nvars)

    for transform in ["fft2d", "dct2d"]:
        loss = get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.spectral.SpectralCRPSLoss",
                    "transform": transform,
                    "x_dim": x_dim,
                    "y_dim": y_dim,
                    "cutoff_ratio": 0.5,
                    "scalers": [],
                },
            ),
        )

        out = loss(pred, target, squash=False)
        assert out.shape == (nvars,), f"{transform}: per-variable CRPS expected"
        out_total = loss(pred, target, squash=True)
        assert out_total.numel() == 1, f"{transform}: scalar CRPS expected"
