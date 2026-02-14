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

import numpy as np
import omegaconf
import pytest
import torch
import yaml

from anemoi.training.diagnostics.callbacks import _get_progress_bar_callback
from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.diagnostics.callbacks.evaluation import RolloutEval
from anemoi.training.diagnostics.callbacks.evaluation import RolloutEvalEns
from anemoi.training.diagnostics.callbacks.plot_ens import EnsemblePlotMixin
from anemoi.training.diagnostics.callbacks.plot_ens import PlotEnsSample
from anemoi.training.diagnostics.callbacks.plot_ens import PlotHistogram
from anemoi.training.diagnostics.callbacks.plot_ens import PlotSample
from anemoi.training.diagnostics.callbacks.plot_ens import PlotSpectrum

NUM_FIXED_CALLBACKS = 3  # ParentUUIDCallback, CheckVariableOrder, RegisterMigrations

default_config = """
training:
  model_task: anemoi.training.train.tasks.GraphEnsForecaster

diagnostics:
  callbacks: []

  plot:
    enabled: False
    focus_areas: null
    callbacks: []

  debug:
    # this will detect and trace back NaNs / Infs etc. but will slow down training
    anomaly_detection: False

  enable_progress_bar: False
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
    pl_module.allgather_batch.side_effect = lambda x, _y, _z: x

    # Mock ensemble output
    loss = torch.tensor(0.5)
    y_preds = [{"data": torch.randn(2, 3, 4, 5)}, {"data": torch.randn(2, 3, 4, 5)}]
    output = [loss, y_preds]

    # Mock batch
    batch = {"data": torch.randn(2, 10, 4, 5)}

    processed_batch, processed_output = mixin._handle_ensemble_batch_and_output(pl_module, output, batch)

    # Check that batch is returned
    assert torch.equal(processed_batch["data"], batch["data"])
    # Check that output is restructured as [loss, y_preds]
    assert len(processed_output) == 2
    assert torch.equal(processed_output[0], loss)
    assert len(processed_output[1]) == 2


def test_ensemble_plot_mixin_process():
    """Test EnsemblePlotMixin.process method."""
    mixin = EnsemblePlotMixin()
    mixin.sample_idx = 0
    mixin.latlons = {"data": np.zeros((100, 2))}

    # Mock lightning module
    pl_module = MagicMock()
    pl_module.n_step_input = 2
    pl_module.rollout = 3
    pl_module.n_step_output = 1
    data_indices = MagicMock()
    data_indices.data.output.full = slice(None)
    pl_module.data_indices = {"data": data_indices}

    # Mock config
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    # Create test tensors
    # batch: bs, input_steps + forecast_steps, ens, latlon, nvar
    batch = {"data": torch.randn(2, 6, 1, 100, 5)}
    # input_tensor: bs, rollout + 1, ens, latlon, nvar
    data_tensor = torch.randn(2, 4, 1, 100, 5)
    # loss: 1, y_preds: bs, multi-out, ens, latlon, nvar
    y_preds = [{"data": torch.randn(2, 1, 1, 100, 5)} for _ in range(3)]
    outputs = [torch.tensor(0.5), y_preds]

    # Mock post_processors
    mock_post_processors = MagicMock()
    mock_post_processors.return_value = data_tensor
    # tensor after post_processors: bs, multi-out, ensemble, latlon, nvar
    mock_post_processors.side_effect = [
        data_tensor,
        torch.randn(2, 1, 1, 100, 5),
        torch.randn(2, 1, 1, 100, 5),
        torch.randn(2, 1, 1, 100, 5),
    ]
    mock_post_processors.cpu.return_value = mock_post_processors
    pl_module.model.post_processors = {"data": mock_post_processors}

    # Mock output_mask.apply as identity
    mock_output_mask = MagicMock()
    mock_output_mask.apply.side_effect = lambda x, **_kwargs: x
    pl_module.output_mask = {"data": mock_output_mask}

    # Set post_processors on the mixin instance
    mixin.post_processors = {"data": mock_post_processors}

    if config["training"]["model_task"] == "anemoi.training.train.tasks.GraphInterpolator":
        output_times = (len(config.training.explicit_times.target), "time_interp")
    else:
        output_times = (getattr(pl_module, "rollout", 0), "forecast")

    data, result_output_tensor = mixin.process(pl_module, "data", outputs, batch, output_times=output_times, members=0)

    # Check instantiation
    assert data is not None
    assert result_output_tensor is not None

    # Check dimensions
    assert data.shape == (4, 1, 100, 5), f"Expected data shape (4, 1, 100, 5), got {data.shape}"
    assert result_output_tensor.shape == (
        3,
        1,
        100,
        5,
    ), f"Expected output_tensor shape (3, 1, 100, 5), got {result_output_tensor.shape}"


def test_rollout_eval_ens_eval():
    """Test RolloutEvalEns._eval method."""
    config = omegaconf.OmegaConf.create({})
    callback = RolloutEvalEns(config, rollout=2, every_n_batches=1)

    # Mock pl_module
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")
    pl_module.n_step_input = 1
    pl_module._rollout_step.return_value = [
        (torch.tensor(0.1), {"metric1": torch.tensor(0.2)}, None, None),
        (torch.tensor(0.15), {"metric1": torch.tensor(0.25)}, None, None),
    ]

    # Mock batch (bs, ms, nens_per_device, latlon, nvar)
    batch = {"data": torch.randn(2, 4, 4, 10, 5)}

    with patch.object(callback, "_log") as mock_log:
        callback._eval(pl_module, batch)

        #  Check for output
        mock_log.assert_called_once()
        args = mock_log.call_args[0]
        assert args[1].item() == pytest.approx(0.125)  # (0.1 + 0.15) / 2
        assert args[2]["metric1"].item() == pytest.approx(0.25)  # Last metric value
        assert args[3] == 2  # batch size


def test_rollout_eval_handles_dict_batch():
    """Test RolloutEval._eval with a dict batch (multi-dataset style)."""
    config = omegaconf.OmegaConf.create({})
    callback = RolloutEval(config, rollout=2, every_n_batches=1)

    # Mock pl_module
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")
    pl_module.n_step_input = 1
    pl_module.n_step_output = 1
    pl_module._rollout_step.return_value = [
        (torch.tensor(0.1), {"metric1": torch.tensor(0.2)}, None),
        (torch.tensor(0.15), {"metric1": torch.tensor(0.25)}, None),
    ]

    # Mock batch (bs, ms, ens, latlon, nvar)
    batch = {"data": torch.randn(2, 4, 1, 10, 5)}

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
                    "focus_areas": {},
                    "datashader": False,
                    "asynchronous": False,
                    "frequency": {"batch": 1},
                },
            },
            "data": {"diagnostic": None},
            "system": {
                "output": {"root": "path_to_output", "plots": "plot"},
            },
            "dataloader": {"read_group_size": 1},
        },
    )

    # Test plotting class instantiation
    plot_ens_sample = PlotEnsSample(
        config=config,
        sample_idx=0,
        parameters=["temperature", "pressure"],
        accumulation_levels_plot=[0.1, 0.5, 0.9],
        output_steps=1,
    )
    assert plot_ens_sample is not None

    plot_sample = PlotSample(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        accumulation_levels_plot=[0.5],
        output_steps=1,
    )
    assert plot_sample is not None

    plot_spectrum = PlotSpectrum(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        output_steps=1,
    )
    assert plot_spectrum is not None

    plot_histogram = PlotHistogram(
        config=config,
        sample_idx=0,
        parameters=["temperature"],
        output_steps=1,
    )
    assert plot_histogram is not None


def test_plot_loss_gathers_nan_mask_weights_from_nested_losses(monkeypatch):
    from omegaconf import DictConfig

    import anemoi.training.diagnostics.callbacks.plot as plot_mod
    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.losses.loss import get_loss_function

    data_indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": []}), {"a": 0, "b": 1})
    combined_loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.CombinedLoss",
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["nan_mask_weights"]},
                    {"_target_": "anemoi.training.losses.MAELoss", "scalers": ["nan_mask_weights"]},
                ],
                "loss_weights": [1.0, 1.0],
                "scalers": ["*"],
            },
        ),
        scalers={"nan_mask_weights": ((0, 3, 4), torch.ones(1, 3, 2))},
        data_indices=data_indices,
    )

    callback = plot_mod.PlotLoss.__new__(plot_mod.PlotLoss)
    callback.every_n_batches = 1
    callback.dataset_names = ["data"]
    callback.parameter_groups = {}

    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.loss = {"data": combined_loss}
    pl_module.grid_dim = -2
    pl_module.grid_indices = {"data": MagicMock()}
    pl_module.allgather_batch.side_effect = lambda tensor, *_args: tensor + 1.0

    super_called = []

    def _stub_super(self, *_args, **_kwargs) -> None:
        del self
        super_called.append(True)

    monkeypatch.setattr(plot_mod.BasePerBatchPlotCallback, "on_validation_batch_end", _stub_super, raising=True)

    callback.on_validation_batch_end(
        trainer=trainer,
        pl_module=pl_module,
        output=(torch.tensor(0.0), []),
        batch={"data": torch.zeros((1, 1, 1, 3, 2))},
        batch_idx=0,
    )

    assert super_called == [True]
    assert pl_module.allgather_batch.call_count == 2
    for child_loss in callback.loss["data"].losses:
        torch.testing.assert_close(child_loss.loss.scaler.nan_mask_weights, torch.full((1, 3, 2), 2.0))


# Progress bar callback tests
progress_bar_config = """
training:
  model_task: anemoi.training.train.tasks.GraphEnsForecaster

diagnostics:
  callbacks: []

  plot:
    enabled: False
    callbacks: []

  debug:
    anomaly_detection: False

  enable_checkpointing: False
  checkpoint:

  log: {}

  enable_progress_bar: True
  progress_bar:
    _target_: pytorch_lightning.callbacks.TQDMProgressBar
    refresh_rate: 1
"""


def test_progress_bar_disabled():
    """Test that no progress bar callback is added when disabled."""
    config = omegaconf.OmegaConf.create(yaml.safe_load(progress_bar_config))
    config.diagnostics.enable_progress_bar = False

    callbacks = _get_progress_bar_callback(config)
    assert len(callbacks) == 0


def test_progress_bar_default():
    """Test that default TQDMProgressBar is used when progress_bar config has no _target_."""
    from pytorch_lightning.callbacks import TQDMProgressBar

    config = omegaconf.OmegaConf.create(yaml.safe_load(progress_bar_config))
    config.diagnostics.progress_bar = None  # No _target_ specified

    callbacks = _get_progress_bar_callback(config)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], TQDMProgressBar)


def test_progress_bar_custom():
    """Test that custom progress bar can be instantiated via _target_."""
    from pytorch_lightning.callbacks import RichProgressBar

    config = omegaconf.OmegaConf.create(yaml.safe_load(progress_bar_config))
    config.diagnostics.progress_bar = {
        "_target_": "pytorch_lightning.callbacks.RichProgressBar",
    }

    callbacks = _get_progress_bar_callback(config)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], RichProgressBar)
