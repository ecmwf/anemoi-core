# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN201

from unittest.mock import MagicMock

import numpy as np
import omegaconf
import torch

from anemoi.training.diagnostics.callbacks.plot_adapter import EnsemblePlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_ens import PlotEnsSample

NUM_FIXED_CALLBACKS = 3  # ParentUUIDCallback, CheckVariableOrder, RegisterMigrations

default_config = """
diagnostics:
  callbacks: []

  plot:
    enabled: False
    callbacks: []

  debug:
    # this will detect and trace back NaNs / Infs etc. but will slow down training
    anomaly_detection: False

  enable_checkpointing: False
  checkpoint:

  log: {}
"""


# Ensemble adapter tests
def test_ensemble_plot_adapter_is_ensemble():
    """Test EnsemblePlotAdapter.is_ensemble property."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    adapter = EnsemblePlotAdapter(inner)
    assert adapter.is_ensemble is True
    assert inner.is_ensemble is False


def test_ensemble_plot_adapter_select_members():
    """Test EnsemblePlotAdapter.select_members method."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    adapter = EnsemblePlotAdapter(inner)

    tensor = torch.randn(2, 3, 4, 100, 5)  # (batch, steps, members, grid, vars)

    # Select single member
    result = adapter.select_members(tensor, members=0)
    assert result.shape == (2, 3, 1, 100, 5)

    # Select multiple members
    result = adapter.select_members(tensor, members=[0, 2])
    assert result.shape == (2, 3, 2, 100, 5)

    # Select all members (None)
    result = adapter.select_members(tensor, members=None)
    assert result.shape == (2, 3, 4, 100, 5)
    assert torch.equal(result, tensor)


def test_ensemble_plot_adapter_prepare_loss_batch():
    """Test EnsemblePlotAdapter.prepare_loss_batch squeezes to member 0."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    adapter = EnsemblePlotAdapter(inner)

    batch = {"data": torch.randn(2, 5, 3, 100, 5)}  # (batch, time, members, grid, vars)
    result = adapter.prepare_loss_batch(batch)

    assert result["data"].shape == (2, 5, 100, 5)
    assert torch.equal(result["data"], batch["data"][:, :, 0, :, :])


def test_ensemble_plot_adapter_delegates_to_inner():
    """Test that EnsemblePlotAdapter delegates iter_plot_samples and other methods to inner."""
    task = MagicMock()
    inner = MagicMock()
    inner._task = task
    inner.get_loss_plot_batch_start.return_value = 42
    inner.prepare_plot_output_tensor.side_effect = lambda x: x

    adapter = EnsemblePlotAdapter(inner)

    assert adapter.get_loss_plot_batch_start(rollout_step=1) == 42
    inner.get_loss_plot_batch_start.assert_called_once_with(rollout_step=1)

    tensor = torch.randn(3, 4)
    adapter.prepare_plot_output_tensor(tensor)
    inner.prepare_plot_output_tensor.assert_called_once_with(tensor)

    data = np.zeros((5, 100, 10))
    output = np.zeros((3, 100, 10))
    list(adapter.iter_plot_samples(data, output))
    inner.iter_plot_samples.assert_called_once_with(data, output)


def test_base_adapter_select_members_is_noop():
    """Test that BasePlotAdapter.select_members is a no-op."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    tensor = torch.randn(2, 3, 100, 5)

    result = inner.select_members(tensor, members=0)
    assert torch.equal(result, tensor)


def test_base_adapter_prepare_loss_batch_is_noop():
    """Test that BasePlotAdapter.prepare_loss_batch is a no-op."""
    task = MagicMock()
    inner = ForecasterPlotAdapter(task)
    batch = {"data": torch.randn(2, 5, 100, 5)}

    result = inner.prepare_loss_batch(batch)
    assert torch.equal(result["data"], batch["data"])


def test_ensemble_plot_ens_sample_instantiation():
    """Test that PlotEnsSample can be instantiated."""
    config = omegaconf.OmegaConf.create(
        {
            "system": {"output": {"plots": "path_to_plots"}},
            "diagnostics": {
                "plot": {
                    "parameters": ["temperature", "pressure"],
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"batch": 1},
                },
            },
            "data": {
                "diagnostic": None,
                "datasets": {"data": {"diagnostic": None}},
            },
            "dataloader": {"read_group_size": 1},
        },
    )

    plot_ens_sample = PlotEnsSample(
        config=config,
        sample_idx=0,
        parameters=["temperature", "pressure"],
        accumulation_levels_plot=[0.1, 0.5, 0.9],
        members=None,
    )
    assert plot_ens_sample is not None
    assert plot_ens_sample.plot_members is None
