# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.training.tasks import Autoencoder
from anemoi.training.tasks import DAForecaster
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler


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


# The adapter's tag counts DA cycles, so with da_cycles=2 forecast step rstep{k} is tagged rstep{k + 2:02d}.
@pytest.mark.parametrize(
    ("steps", "expected"),
    [(None, ["rstep02_out00", "rstep03_out00", "rstep04_out00"]), (["rstep2"], ["rstep04_out00"])],
)
def test_forecaster_adapter_steps_filter(steps: list[str] | None, expected: list[str]) -> None:
    """``steps`` keeps only the listed forecast steps; DA cycles are never plotted."""
    rollout = {"start": 1, "epoch_increment": 1, "maximum": 3}
    task = DAForecaster(
        multistep_input=2,
        multistep_output=1,
        timestep="6H",
        rollout=rollout,
        validation_rollout=3,
        da_cycles=2,
    )
    n_steps = len(task.steps("validation"))
    batch = torch.randn(2 + n_steps, 1, 10, 4)  # (time, ens, grid, vars)
    pred = torch.randn(n_steps, 1, 10, 4)  # (step, out_step, grid, vars)

    suffixes = [suffix for *_, suffix in task._plot_adapter.iter_plot_samples(batch, pred, steps=steps)]
    assert suffixes == expected


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
