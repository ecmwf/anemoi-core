# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MAELoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import WeightedMSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.spectral import SpectralCRPSLoss


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
