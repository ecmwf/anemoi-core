# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spectral-loss config selects native, graphed-native or padded reduced-grid FFTs."""

import pytest
import torch
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.layers.ring_fft import shared_ring_fft
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.training.losses import SpectralCRPSLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.schemas.training import LossSchemas


_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


def _view(data: torch.Tensor) -> GriddedSource:
    """Wrap a test tensor without detaching it from autograd."""
    return GriddedSource(
        name="data",
        values=data,
        variables=[f"v{i}" for i in range(data.shape[-1])],
        statistics={},
        coordinates=torch.zeros(data.shape[_LAYOUT.grid], 2, device=data.device),
        layout=_LAYOUT,
    )


@pytest.mark.parametrize(
    ("backend", "graphed", "device"),
    [
        ("ring", False, "cpu"),
        ("padded", False, "cpu"),
        pytest.param(
            "ring",
            True,
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_spectral_fft_config_matches_native_loss_and_gradient(backend: str, graphed: bool, device: str) -> None:
    config = {
        "_target_": "anemoi.training.losses.SpectralCRPSLoss",
        "transform": "octahedral_sht",
        "nlat": 8,
        "truncation": 2,
        "fft_backend": backend,
        "use_graphed_rfft": graphed,
        "scalers": [],
    }
    validated = TypeAdapter(LossSchemas).validate_python(config)
    assert validated.fft_backend == backend
    loss = get_loss_function(DictConfig(config)).to(device)
    reference = SpectralCRPSLoss(transform="octahedral_sht", nlat=8, truncation=2).to(device)
    points = sum(octahedral_lons_per_lat(8))
    torch.manual_seed(5)
    pred = torch.randn(1, 1, 3, points, 2, device=device, requires_grad=True)
    target = torch.randn(1, 1, 1, points, 2, device=device)
    pred_view, target_view = _view(pred), _view(target)
    try:
        actual, expected = loss(pred_view, target_view), reference(pred_view, target_view)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        got = torch.autograd.grad(actual, pred)[0]
        want = torch.autograd.grad(expected, pred)[0]
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
    finally:
        shared_ring_fft.cache_clear()


@pytest.mark.parametrize(
    "options",
    [
        {"fft_backend": "unknown"},
        {"fft_backend": "padded", "use_graphed_rfft": True},
    ],
)
def test_spectral_schema_rejects_invalid_fft_options(options: dict) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(LossSchemas).validate_python(
            {
                "_target_": "anemoi.training.losses.SpectralCRPSLoss",
                "transform": "octahedral_sht",
                "nlat": 8,
                **options,
            },
        )
