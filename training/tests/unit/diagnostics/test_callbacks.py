# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN001, ANN201

from unittest.mock import MagicMock
from unittest.mock import patch

import omegaconf
import pytest
import torch
import yaml

from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.diagnostics.callbacks.evaluation import RolloutEvalEns
from anemoi.training.diagnostics.callbacks.plot_ens import EnsemblePlotMixin
from anemoi.training.diagnostics.callbacks.plot_ens import PlotEnsSample
from anemoi.training.diagnostics.callbacks.plot_ens import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot_ens import PlotSample
from anemoi.training.diagnostics.callbacks.plot_ens import PlotSpectrum

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


def test_no_extra_callbacks_set():
    # No extra callbacks set
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    callbacks = get_callbacks(config)
    assert len(callbacks) == NUM_FIXED_CALLBACKS  # ParentUUIDCallback, CheckVariableOrder, etc


def test_add_config_enabled_callback():
    # Add logging callback
    config = omegaconf.OmegaConf.create(default_config)
    config.diagnostics.callbacks.append({"log": {"mlflow": {"enabled": True}}})
    callbacks = get_callbacks(config)
    assert len(callbacks) == NUM_FIXED_CALLBACKS + 1


def test_add_callback():
    config = omegaconf.OmegaConf.create(default_config)
    config.diagnostics.callbacks.append(
        {"_target_": "anemoi.training.diagnostics.callbacks.provenance.ParentUUIDCallback"},
    )
    callbacks = get_callbacks(config)
    assert len(callbacks) == NUM_FIXED_CALLBACKS + 1


def test_add_plotting_callback(monkeypatch):
    # Add plotting callback
    import anemoi.training.diagnostics.callbacks.plot as plot

    class PlotLoss:
        def __init__(self, config: omegaconf.DictConfig):
            pass

    monkeypatch.setattr(plot, "PlotLoss", PlotLoss)

    config = omegaconf.OmegaConf.create(default_config)
    config.diagnostics.plot.enabled = True
    config.diagnostics.plot.callbacks = [{"_target_": "anemoi.training.diagnostics.callbacks.plot.PlotLoss"}]
    callbacks = get_callbacks(config)
    assert len(callbacks) == NUM_FIXED_CALLBACKS + 1


# Ensemble callback tests
def test_ensemble_plot_mixin_handle_batch_and_output():
    """Test EnsemblePlotMixin._handle_ensemble_batch_and_output method."""
    mixin = EnsemblePlotMixin()

    # Mock lightning module and allgather_batch method
    pl_module = MagicMock()
    pl_module.allgather_batch.side_effect = lambda x: x

    # Mock ensemble output
    loss = torch.tensor(0.5)
    y_preds = [torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)]
    ens_ic = torch.randn(2, 3)
    output = [loss, y_preds, ens_ic]

    # Mock batch
    batch = [torch.randn(2, 10, 4, 5), torch.randn(2, 10, 4, 5)]

    processed_batch, processed_output = mixin._handle_ensemble_batch_and_output(pl_module, output, batch)

    # Check that batch[0] is returned
    assert torch.equal(processed_batch, batch[0])
    # Check that output is restructured as [loss, y_preds]
    assert len(processed_output) == 2
    assert torch.equal(processed_output[0], loss)
    assert len(processed_output[1]) == 2


def test_ensemble_plot_mixin_process():
    """Test EnsemblePlotMixin.process method."""
    mixin = EnsemblePlotMixin()
    mixin.sample_idx = 0
    mixin.latlons = None

    # Mock lightning module
    pl_module = MagicMock()
    pl_module.multi_step = 2
    pl_module.rollout = 3
    pl_module.data_indices.data.output.full = slice(None)
    pl_module.latlons_data = torch.randn(100, 2)

    # Create test tensors
    # batch: bs, input_steps + forecast_steps, latlon, nvar
    batch = torch.randn(2, 6, 100, 5)
    # input_tensor: bs, rollout + 1, latlon, nvar
    data_tensor = torch.randn(2, 4, 100, 5)
    # loss: 1, y_preds: bs, latlon, nvar
    outputs = [torch.tensor(0.5), [torch.randn(2, 100, 5), torch.randn(2, 100, 5), torch.randn(2, 100, 5)]]

    # Mock post_processors
    mock_post_processors = MagicMock()
    mock_post_processors.return_value = data_tensor
    # tensor after post_processors: bs, ensemble, latlon, nvar
    mock_post_processors.side_effect = [
        data_tensor,
        torch.randn(2, 1, 100, 5),
        torch.randn(2, 1, 100, 5),
        torch.randn(2, 1, 100, 5),
    ]
    mock_post_processors.cpu.return_value = mock_post_processors
    pl_module.model.post_processors = mock_post_processors

    # Mock output_mask.apply as identity
    pl_module.output_mask.apply.side_effect = lambda x, **_kwargs: x

    # Set post_processors on the mixin instance
    mixin.post_processors = mock_post_processors

    data, result_output_tensor = mixin.process(pl_module, outputs, batch, members=0)

    # Check instantiation
    assert data is not None
    assert result_output_tensor is not None

    # Check dimensions
    assert data.shape == (4, 100, 5), f"Expected data shape (4, 100, 5), got {data.shape}"
    assert result_output_tensor.shape == (
        3,
        100,
        5,
    ), f"Expected output_tensor shape (3, 100, 5), got {result_output_tensor.shape}"


def test_rollout_eval_ens_eval():
    """Test RolloutEvalEns._eval method."""
    config = omegaconf.OmegaConf.create({})
    callback = RolloutEvalEns(config, rollout=2, every_n_batches=1)

    # Mock pl_module
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")
    pl_module.multi_step = 1
    pl_module.rollout_step.return_value = [
        (torch.tensor(0.1), {"metric1": torch.tensor(0.2)}, None, None),
        (torch.tensor(0.15), {"metric1": torch.tensor(0.25)}, None, None),
    ]

    # Mock batch (bs, ms, nens_per_device, latlon, nvar)
    batch = torch.randn(2, 4, 4, 10, 5)

    with patch.object(callback, "_log") as mock_log:
        callback._eval(pl_module, batch)

        #  Check for output
        mock_log.assert_called_once()
        args = mock_log.call_args[0]
        assert args[1].item() == pytest.approx(0.125)  # (0.1 + 0.15) / 2
        assert args[2]["metric1"].item() == pytest.approx(0.25)  # Last metric value
        assert args[3] == 2  # batch size


def test_ensemble_plot_callbacks_instantiation():
    """Test that ensemble plot callbacks can be instantiated."""
    config = omegaconf.OmegaConf.create(
        {
            "diagnostics": {
                "plot": {
                    "parameters": ["temperature", "pressure"],
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"batch": 1},
                },
            },
            "data": {"diagnostic": None},
            "hardware": {"paths": {"plots": "path_to_plots"}},
            "dataloader": {"read_group_size": 1},
        },
    )

    # Test plotting class instantiation
    plot_ens_sample = PlotEnsSample(
        config=config,
        sample_idx=0,
        parameters=["temperature", "pressure"],
        accumulation_levels_plot=[0.1, 0.5, 0.9],
    )
    assert plot_ens_sample is not None

    plot_sample = PlotSample(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        accumulation_levels_plot=[0.5],
    )
    assert plot_sample is not None

    plot_spectrum = PlotSpectrum(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
    )
    assert plot_spectrum is not None

    plot_histogram = PlotHistogram(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
    )
    assert plot_histogram is not None
