# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import torch

from anemoi.training.tasks import SpatialDownscaler


def test_spatial_downscaler_offsets() -> None:
    """SpatialDownscaler has a single input and output offset at t=0."""
    task = SpatialDownscaler()
    assert task._input_offsets == [datetime.timedelta(0)]
    assert task._output_offsets == [datetime.timedelta(0)]


def test_spatial_downscaler_default_dataset_names() -> None:
    task = SpatialDownscaler()
    assert task.source_dataset == "in_lres"
    assert task.target_dataset == "out_hres"


def test_spatial_downscaler_custom_dataset_names() -> None:
    task = SpatialDownscaler(source_dataset="lres", target_dataset="hres")
    assert task.source_dataset == "lres"
    assert task.target_dataset == "hres"


def test_prepare_inputs_upsamples_source() -> None:
    """prepare_inputs calls model.model.residual[source_dataset] and adds ensemble dim."""
    task = SpatialDownscaler(source_dataset="in_lres", target_dataset="out_hres")

    batch_size, n_time, n_grid, n_vars = 2, 1, 10, 3
    x_lres = torch.randn(batch_size, n_time, n_grid, n_vars)
    # InterpolationConnection selects a single timestep, so it returns (batch, hres_grid, vars)
    n_grid_hres = 20
    x_upsampled_raw = torch.randn(batch_size, n_grid_hres, n_vars)

    mock_interpolator = MagicMock(return_value=x_upsampled_raw)
    mock_model = MagicMock()
    mock_model.model.residual = {"in_lres": mock_interpolator}

    x_in_hres = torch.randn(batch_size, n_time, 1, n_grid_hres, 4)
    x = {"in_lres": x_lres, "in_hres": x_in_hres}

    result = task.prepare_inputs(x, mock_model, model_comm_group=None)

    # Interpolator called with correct args
    mock_interpolator.assert_called_once_with(x_lres, grid_shard_shapes=None, model_comm_group=None)

    # Result has time and ensemble dims added: (batch, time=1, ensemble=1, grid, vars)
    assert result["in_lres"].shape == (batch_size, 1, 1, n_grid_hres, n_vars)
    # Other datasets untouched
    assert result["in_hres"] is x_in_hres


def test_prepare_inputs_preserves_other_datasets() -> None:
    """prepare_inputs does not modify datasets other than source_dataset."""
    task = SpatialDownscaler()
    x_lres = torch.zeros(1, 1, 5, 2)
    x_hres = torch.ones(1, 1, 1, 10, 4)
    x_out = torch.full((1, 1, 1, 10, 2), 7.0)

    mock_model = MagicMock()
    mock_model.model.residual = {"in_lres": MagicMock(return_value=torch.zeros(1, 10, 2))}

    result = task.prepare_inputs(
        {"in_lres": x_lres, "in_hres": x_hres, "out_hres": x_out},
        mock_model,
    )
    assert result["in_hres"] is x_hres
    assert result["out_hres"] is x_out


def test_get_timestep_for_metadata() -> None:
    task = SpatialDownscaler()
    assert task._get_timestep_for_metadata() == "0H"
