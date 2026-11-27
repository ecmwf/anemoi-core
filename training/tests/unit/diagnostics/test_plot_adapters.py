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

from anemoi.training.diagnostics.callbacks.plot_adapter import EnsemblePlotAdapterWrapper
from anemoi.training.tasks import Autoencoder
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


def test_forecaster_plot_steps_cover_every_rollout_output() -> None:
    """Forecaster figures show the latest input next to each output of each validation rollout step."""
    task = Forecaster(
        multistep_input=2,
        multistep_output=2,
        timestep="6H",
        rollout={"start": 1, "epoch_increment": 0, "maximum": 1},
        validation_rollout=2,
    )

    steps = task._plot_adapter.plot_steps()

    input_index = task.get_batch_input_indices()[-1]
    expected = [
        (input_index, task.get_batch_output_indices(rollout_step=rollout_step)[out_step], rollout_step, out_step)
        for rollout_step in range(2)
        for out_step in range(2)
    ]
    assert [(s.input_index, s.target_index, s.prediction_step, s.output_step) for s in steps] == expected
    assert [s.tag for s in steps] == ["rstep00_out00", "rstep00_out01", "rstep01_out00", "rstep01_out01"]


def test_temporal_downscaler_plot_steps_one_per_output_time() -> None:
    """Temporal downscaler figures show the first input next to each output time in turn."""
    task = TemporalDownscaler(input_timestep="6H", output_timestep="1H")

    steps = task._plot_adapter.plot_steps()

    # Batch times are 0H..6H: inputs at positions 0 and 6, outputs at positions 1..5.
    assert [s.input_index for s in steps] == [0] * 5
    assert [s.target_index for s in steps] == [1, 2, 3, 4, 5]
    assert [s.prediction_step for s in steps] == [0] * 5
    assert [s.output_step for s in steps] == [0, 1, 2, 3, 4]
    assert [s.tag for s in steps] == ["istep01", "istep02", "istep03", "istep04", "istep05"]


def test_temporal_downscaler_samples_follow_plot_steps() -> None:
    """Each gridded sample pairs the target time and the prediction named by its plot step."""
    task = TemporalDownscaler(input_timestep="6H", output_timestep="2H")
    adapter = task._plot_adapter

    grid_size, num_vars = 10, 3
    # Every batch time holds its own position, every prediction its output step plus 100.
    batch = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 1, grid_size, num_vars)
    pred = (100 + torch.arange(2, dtype=torch.float32)).view(2, 1, 1, 1).expand(2, 1, grid_size, num_vars)

    samples = list(adapter.iter_plot_samples(batch, pred))

    assert [suffix for *_, suffix in samples] == [s.tag for s in adapter.plot_steps()]
    for (x, y_true, y_pred, _), step in zip(samples, adapter.plot_steps(), strict=True):
        assert torch.all(x == step.input_index)
        assert torch.all(y_true == step.target_index)
        assert torch.all(y_pred == 100 + step.output_step)


def test_ensemble_wrapper_uses_inner_plot_steps() -> None:
    """The ensemble wrapper shows the same times as the task adapter it wraps."""
    task = TemporalDownscaler(input_timestep="6H", output_timestep="2H")

    wrapper = EnsemblePlotAdapterWrapper(task._plot_adapter)

    assert wrapper.plot_steps() == task._plot_adapter.plot_steps()
