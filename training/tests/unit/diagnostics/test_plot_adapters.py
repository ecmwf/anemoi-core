# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import MethodType
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportSpatialDownscalerModelEncProcDec
from anemoi.training.tasks import Autoencoder
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.tasks.spatial_downscaler import SpatialDownscaler


@pytest.mark.parametrize("rollout_steps", [1, 6, 12])
def test_forecaster_adapter(rollout_steps: int) -> None:
    """Forecaster plot_adapter."""
    rollout = {"start": 1, "epoch_increment": 1, "maximum": rollout_steps}
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6H", rollout=rollout)

    adapter = task._plot_adapter

    # Example: [-6H, 0H] input, [6H] output. 1 ens member ...
    grid_size = 1000
    num_vars = 12
    batch = torch.randn(4, 1, grid_size, num_vars)  # (time, ens, grid, vars)
    pred = torch.randn(2, 1, grid_size, num_vars)  # (time, ens, grid, vars)

    x, y_true, y_pred, suffix = next(adapter.iter_plot_samples(batch, pred))
    assert isinstance(x, torch.Tensor) and x.shape == (grid_size, num_vars)
    assert isinstance(y_true, torch.Tensor) and y_true.shape == (grid_size, num_vars)
    assert isinstance(y_pred, torch.Tensor) and y_pred.shape == (grid_size, num_vars)
    assert isinstance(suffix, str) and suffix.startswith("rstep")


def test_temporal_downscaler_adapter() -> None:
    """TemporalDownscaler plot_adapter."""
    task = TemporalDownscaler(input_timestep="6H", output_timestep="3H", output_left_boundary=True)

    adapter = task._plot_adapter

    # Example: [0H, 6H] input, [0H, 2H, 4H] output. 1 ens member ...
    grid_size = 1000
    num_vars = 12
    batch = torch.randn(4, 1, grid_size, num_vars)  # (time, ens, grid, vars)
    pred = torch.randn(task.num_output_timesteps, 1, grid_size, num_vars)  # (time, ens, grid, vars)

    x, y_true, y_pred, suffix = next(adapter.iter_plot_samples(batch, pred))
    assert isinstance(x, torch.Tensor) and x.shape == (grid_size, num_vars)
    assert isinstance(y_true, torch.Tensor) and y_true.shape == (grid_size, num_vars)
    assert isinstance(y_pred, torch.Tensor) and y_pred.shape == (grid_size, num_vars)
    assert isinstance(suffix, str) and suffix.startswith("istep")


def test_autoencoder_adapter() -> None:
    """Autoencoder plot_adapter."""
    task = Autoencoder()

    adapter = task._plot_adapter

    # Example: [0H, 6H] input, [0H, 2H, 4H] output. 1 ens member ...
    grid_size = 1000
    num_vars = 12
    batch = torch.randn(1, 1, grid_size, num_vars)  # (time, ens, grid, vars)
    pred = torch.randn(1, 1, grid_size, num_vars)  # (time, ens, grid, vars)

    x, y_true, y_pred, suffix = next(adapter.iter_plot_samples(batch, pred))
    assert isinstance(x, torch.Tensor) and x.shape == (grid_size, num_vars)
    assert isinstance(y_true, torch.Tensor) and y_true.shape == (grid_size, num_vars)
    assert isinstance(y_pred, torch.Tensor) and y_pred.shape == (grid_size, num_vars)
    assert isinstance(suffix, str) and suffix.startswith("recon")


def _spatial_downscaler_task() -> SpatialDownscaler:
    return SpatialDownscaler(
        input_datasets=["in_lres"],
        target_datasets=["out_hres"],
        input_offsets=["0H", "6H"],
        output_offsets=["0H", "6H"],
    )


def test_spatial_downscaler_adapter_pairs_each_output_step_with_the_reference_at_the_same_time() -> None:
    """X is the projected reference at the output time, not the target's own (truth) state."""
    adapter = _spatial_downscaler_task()._plot_adapter

    grid_size, num_vars = 100, 4
    batch = torch.randn(2, 1, grid_size, num_vars)  # (time, ens, grid, vars): truth of the target
    reference = torch.randn(2, 1, grid_size, num_vars)  # (time, ens, grid, vars): projected low-res
    pred = adapter.prepare_plot_output_tensor(torch.randn(1, 2, 1, grid_size, num_vars))

    samples = list(adapter.iter_plot_samples(batch, pred, input_data=reference))

    assert [suffix for *_, suffix in samples] == ["out00", "out01"]
    for step, (x, y_true, y_pred, _) in enumerate(samples):
        torch.testing.assert_close(x, reference[step].squeeze())
        torch.testing.assert_close(y_true, batch[step].squeeze())
        torch.testing.assert_close(y_pred, pred[step].squeeze())


def _identity_post_processor(
    x: torch.Tensor,
    in_place: bool = False,
    data_index: torch.Tensor | None = None,
    skip_imputation: bool = False,
) -> torch.Tensor:
    del in_place, data_index, skip_imputation
    return x.clone()


def _downscaler_pl_module(task: SpatialDownscaler) -> MagicMock:
    """pl_module whose model builds the reference with the real ``reference_on_target_grid``.

    The reference ``in_lres`` and the target ``out_hres`` list their variables in different orders,
    and the target has a diagnostic ``tp`` without low-res counterpart.
    """
    in_lres = SimpleNamespace(
        data=SimpleNamespace(input=SimpleNamespace(full=torch.tensor([0, 1, 2]))),
        model=SimpleNamespace(input=SimpleNamespace(name_to_index={"10u": 0, "2t": 1, "lsm": 2})),
    )
    out_hres = SimpleNamespace(
        model=SimpleNamespace(output=SimpleNamespace(name_to_index={"2t": 0, "10u": 1, "tp": 2})),
    )
    data_indices = {"in_lres": in_lres, "out_hres": out_hres}
    model = SimpleNamespace(
        target_datasets=["out_hres"],
        residual_reference={"out_hres": "in_lres"},
        data_indices=data_indices,
    )
    model.reference_on_target_grid = MethodType(
        AnemoiTransportSpatialDownscalerModelEncProcDec.reference_on_target_grid,
        model,
    )
    pl_module = MagicMock()
    pl_module.task = task
    pl_module.data_indices = data_indices
    pl_module.model.model = model
    return pl_module


def test_spatial_downscaler_adapter_prepares_the_reference_in_the_target_output_layout() -> None:
    task = _spatial_downscaler_task()
    adapter = task._plot_adapter
    pl_module = _downscaler_pl_module(task)

    batch = {
        "in_lres": torch.randn(2, 2, 1, 10, 3),
        "out_hres": torch.randn(2, 2, 1, 10, 4),
    }
    post_processors = {"in_lres": _identity_post_processor, "out_hres": _identity_post_processor}

    plot_input = adapter.prepare_plot_input(pl_module, "out_hres", batch, post_processors)

    assert plot_input.shape == (2, 2, 1, 10, 3)
    torch.testing.assert_close(plot_input[..., 0], batch["in_lres"][..., 1])  # 2t
    torch.testing.assert_close(plot_input[..., 1], batch["in_lres"][..., 0])  # 10u
    assert torch.all(plot_input[..., 2] == 0)  # tp: no low-res counterpart
