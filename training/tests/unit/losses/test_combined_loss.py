# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from unittest.mock import MagicMock

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses import SpectralL2Loss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.multiscale import MultiscaleLossWrapper


class FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def test_combined_loss() -> None:
    """Test the combined loss function."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
                "scalers": ["test"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"test": (-1, torch.ones(2))},
    )
    assert isinstance(loss.losses[0], MSELoss)
    assert "test" in loss.losses[0].scaler

    assert isinstance(loss.losses[1], MAELoss)
    assert "test" in loss.losses[1].scaler


def test_combined_loss_invalid_loss_weights() -> None:
    """Test the combined loss function with invalid loss weights."""
    with pytest.raises(InstantiationException):
        get_loss_function(
            DictConfig(
                {
                    "_target_": "anemoi.training.losses.combined.CombinedLoss",
                    "losses": [
                        {"_target_": "anemoi.training.losses.MSELoss"},
                        {"_target_": "anemoi.training.losses.MAELoss"},
                    ],
                    "scalers": ["test"],
                    "loss_weights": [1.0, 0.5, 1],
                },
            ),
            scalers={"test": (-1, torch.ones(2))},
        )


def test_combined_loss_equal_weighting() -> None:
    """Test equal weighting when not given."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss"},
                    {"_target_": "anemoi.training.losses.MAELoss"},
                ],
            },
        ),
        scalers={},
    )
    assert all(weight == 1.0 for weight in loss.loss_weights)


def test_combined_loss_seperate_scalers() -> None:
    """Test that scalers are passed to the correct loss function."""
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["test"]},
                    {"_target_": "anemoi.training.losses.MAELoss", "scalers": ["test2"]},
                ],
                "scalers": ["test", "test2"],
                "loss_weights": [1.0, 0.5],
            },
        ),
        scalers={"test": (-1, torch.ones(2)), "test2": (-1, torch.ones(2))},
    )
    assert isinstance(loss, CombinedLoss)

    assert isinstance(loss.losses[0], MSELoss)
    assert "test" in loss.losses[0].scaler
    assert "test2" not in loss.losses[0].scaler

    assert isinstance(loss.losses[1], MAELoss)
    assert "test" not in loss.losses[1].scaler
    assert "test2" in loss.losses[1].scaler


def test_combined_loss_propagates_needs_shard_layout_info() -> None:
    loss = CombinedLoss(
        MultiscaleLossWrapper(
            per_scale_loss=MSELoss(),
            weights=[1.0],
            keep_batch_sharded=True,
        ),
    )

    assert loss.needs_shard_layout_info is True


def test_combined_loss_mixed_children_filter_shard_layout_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    pred = torch.zeros((1, 1, 1, 2, 1))
    target = torch.zeros((1, 1, 2, 1))
    grid_shard_shapes = [(1, 2, 1), (1, 2, 1)]
    weights = torch.ones((1, 1, 1, 1, 1))
    group = FakeGroup(size=2)

    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=MSELoss(),
        weights=[1.0],
        keep_batch_sharded=True,
    )
    prepare_for_smoothing = MagicMock(return_value=(pred, target, grid_shard_shapes, grid_shard_shapes))
    monkeypatch.setattr(multiscale_loss, "_prepare_for_smoothing", prepare_for_smoothing)
    monkeypatch.setattr("anemoi.training.losses.multiscale.gather_channels", lambda x, *_args: x)
    monkeypatch.setattr("anemoi.training.losses.base.reduce_tensor", lambda x, *_args: x)

    loss = CombinedLoss(
        multiscale_loss,
        WeightedMSELoss(),
    )

    result = loss(
        pred,
        target,
        weights=weights,
        group=group,
        grid_dim=-2,
        grid_shard_shapes=grid_shard_shapes,
    )

    assert result.shape == (1,)
    prepare_for_smoothing.assert_called_once_with(pred, target, group, -2, grid_shard_shapes)


def test_iter_leaf_losses_combined() -> None:
    """Test that iter_leaf_losses on a CombinedLoss yields the sub-losses."""
    mse = MSELoss()
    mae = MAELoss()
    combined = CombinedLoss(losses=[mse, mae], loss_weights=[1.0, 1.0])

    leaves = list(combined.iter_leaf_losses())
    assert len(leaves) == 2
    assert leaves[0] is mse
    assert leaves[1] is mae


def test_combined_loss_with_spectral_crps_backward() -> None:
    # Use a tiny regular 2D field so we can use FFT2D-based spectral loss without extra assets.
    batch = 2
    ensemble = 4  # SpectralCRPSLoss is intended for ensemble training
    y_dim = 8
    x_dim = 6
    points = x_dim * y_dim
    variables = 3

    # Match the typical tensor layout used by Anemoi losses:
    pred = torch.randn(batch, 1, ensemble, points, variables, requires_grad=True)
    target = torch.randn(batch, 1, 1, points, variables)  # allow broadcasting over ensemble if supported

    # Node weights are commonly required by the weighted loss base class; keep them neutral.
    node_weights = torch.ones(points)

    mse = WeightedMSELoss()
    spectral = SpectralCRPSLoss(node_weights=node_weights, transform="fft2d", x_dim=x_dim, y_dim=y_dim)

    loss = CombinedLoss(
        losses=[mse, spectral],
        loss_weights=[1.0, 0.25],
    )

    out = loss(pred, target)
    assert out.ndim == 0
    assert torch.isfinite(out).all()

    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_combined_loss_with_spectral_l2_loss_backward() -> None:

    def _octahedral_expected_points(nlat: int) -> int:
        half = [4 * (i + 1) + 16 for i in range(nlat // 2)]
        nlon = half + half[::-1]
        return int(sum(nlon))

    nlat = 8
    nvars = 3
    expected_points = _octahedral_expected_points(nlat)
    # Match the typical tensor layout used by Anemoi losses:
    pred = torch.zeros((2, 1, 1, expected_points, nvars), requires_grad=True)
    target = torch.zeros_like(pred)

    mse = WeightedMSELoss()
    spectral = SpectralL2Loss(
        transform="octahedral_sht",
        nlat=nlat,
    )

    loss = CombinedLoss(
        losses=[mse, spectral],
        loss_weights=[1.0, 0.25],
    )

    out = loss(pred, target)
    assert out.ndim == 0
    assert torch.isfinite(out).all()

    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
