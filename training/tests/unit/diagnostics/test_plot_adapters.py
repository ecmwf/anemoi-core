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
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler


@pytest.mark.parametrize("rollout_steps", [1, 6, 12])
def test_forecaster_adapter(rollout_steps: int):
    """Forecaster plot_adapter: output_times from task.rollout, get_init_step() == -1."""
    rollout = {"start": 1, "epoch_increment": 1, "maximum": rollout_steps}
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6H", rollout=rollout)

    adapter = task._plot_adapter

    assert adapter.output_times == 1
    assert adapter.get_init_step() == -1

    # Example: [-6H, 0H] input, [6H] output. 1 ens member ...
    batch = torch.randn(4, 1, 1000, 12)  # (time, ens, grid, vars)
    pred = torch.randn(2, 1, 1000, 12) # (time, ens, grid, vars)

    x, y_true, y_pred, suffix = next(adapter.iter_plot_samples(batch, pred, None))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(suffix, str)


def test_temporal_downscaler_adapter():
    """TemporalDownscaler plot_adapter.

    output_times == len(interp_times), get_init_step() == 0.
    """
    task = TemporalDownscaler(input_timestep="6H", output_timestep="3H", output_left_boundary=True)

    adapter = task._plot_adapter

    assert adapter.output_times == 2
    assert adapter.get_init_step() == 0

    # Example: [0H, 6H] input, [0H, 2H, 4H] output. 1 ens member ...
    batch = torch.randn(4, 1, 1000, 12)  # (time, ens, grid, vars)
    pred = torch.randn(adapter.output_times, 1, 1000, 12) # (time, ens, grid, vars)

    x, y_true, y_pred, suffix = next(adapter.iter_plot_samples(batch, pred, None))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(suffix, str)


def test_autoencoder_adapter():
    """Autoencoder plot_adapter: output_times == 1, get_init_step() == 0."""
    task = Autoencoder()

    adapter = task._plot_adapter

    assert adapter.output_times == 1
    assert adapter.get_init_step() == 0