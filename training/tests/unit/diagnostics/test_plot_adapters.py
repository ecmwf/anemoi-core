# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.tasks import Autoencoder
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler


@pytest.mark.parametrize("rollout_steps", [1, 6, 12])
def test_output_times_and_get_init_step_forecaster(rollout_steps: int):
    """Forecaster plot_adapter: output_times from task.rollout, get_init_step() == -1."""
    rollout = {"start": 1, "epoch_increment": 1, "maximum": rollout_steps}
    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6H", rollout=rollout)

    adapter = task._plot_adapter

    assert adapter.output_times == 1
    assert adapter.get_init_step() == -1


def test_output_times_and_get_init_step_temporal_downscaler():
    """TemporalDownscaler plot_adapter.

    output_times == len(interp_times), get_init_step() == 0.
    """
    task = TemporalDownscaler(input_timestep="6H", output_timestep="3H", output_left_boundary=True)

    adapter = task._plot_adapter

    assert adapter.output_times == 2
    assert adapter.get_init_step() == 0


def test_output_times_and_get_init_step_autoencoder():
    """Autoencoder plot_adapter: output_times == 1, get_init_step() == 0."""
    task = Autoencoder()

    adapter = task._plot_adapter

    assert adapter.output_times == 1
    assert adapter.get_init_step() == 0
