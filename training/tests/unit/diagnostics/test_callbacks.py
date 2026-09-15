# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: ANN001, ANN201

from unittest.mock import MagicMock

import omegaconf
import torch
import yaml

from anemoi.training.diagnostics.callbacks import CallbacksContext
from anemoi.training.diagnostics.callbacks import _get_progress_bar_callback
from anemoi.training.diagnostics.callbacks import get_callbacks
from anemoi.training.diagnostics.callbacks.per_timestep_metrics import PerTimestepMetrics

NUM_FIXED_CALLBACKS = 3  # ParentUUIDCallback, CheckVariableOrder, RegisterMigrations

default_config = """
training:
  method:
    _target_: anemoi.training.train.methods.EnsembleTraining
  multistep_input : 1

diagnostics:
  callbacks: []

  plot:
    enabled: False
    datashader: True
    projection_kind: equirectangular
    asynchronous: True
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
    context = CallbacksContext(
        diagnostics=config.diagnostics,
        checkpoints_output=omegaconf.OmegaConf.create({"root": ".test_checkpoints"}),
        plots_output=None,
        wandb_enabled=False,
        mlflow_enabled=False,
    )
    callbacks = get_callbacks(context)
    assert len(callbacks) == NUM_FIXED_CALLBACKS  # ParentUUIDCallback, CheckVariableOrder, etc


def test_add_config_enabled_callback():
    # Add logging callback (mlflow enabled triggers LearningRateMonitor)
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    context = CallbacksContext(
        diagnostics=config.diagnostics,
        checkpoints_output=omegaconf.OmegaConf.create({"root": ".test_checkpoints"}),
        plots_output=None,
        wandb_enabled=False,
        mlflow_enabled=True,
    )
    callbacks = get_callbacks(context)
    assert len(callbacks) == NUM_FIXED_CALLBACKS + 1


def test_add_callback():
    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.diagnostics.callbacks.append(
        {"_target_": "anemoi.training.diagnostics.callbacks.provenance.ParentUUIDCallback"},
    )
    context = CallbacksContext(
        diagnostics=config.diagnostics,
        checkpoints_output=omegaconf.OmegaConf.create({"root": ".test_checkpoints"}),
        plots_output=None,
        wandb_enabled=False,
        mlflow_enabled=False,
    )
    callbacks = get_callbacks(context)
    assert len(callbacks) == NUM_FIXED_CALLBACKS + 1


def test_user_callback_instantiated_via_hydra_interpolation():
    """A callback in diagnostics.callbacks is instantiated with values from the Hydra tree.

    This verifies that callbacks receive config values via Hydra interpolation.
    """
    config = omegaconf.OmegaConf.create(
        yaml.safe_load(
            default_config + """
task:
  validation_rollout: 3
""",
        ),
    )
    config.diagnostics.callbacks.append(
        {
            "_target_": "anemoi.training.diagnostics.callbacks.per_timestep_metrics.PerTimestepMetrics",
            "every_n_batches": "${task.validation_rollout}",
        },
    )
    # Pass config.diagnostics — it keeps its parent reference to the root config,
    # so ${task.validation_rollout} resolves correctly during instantiate().
    context = CallbacksContext(
        diagnostics=config.diagnostics,
        checkpoints_output=omegaconf.OmegaConf.create({"root": ".test_checkpoints"}),
        plots_output=None,
        wandb_enabled=False,
        mlflow_enabled=False,
    )
    callbacks = get_callbacks(context)
    user_callbacks = [cb for cb in callbacks if isinstance(cb, PerTimestepMetrics)]
    assert len(user_callbacks) == 1
    assert user_callbacks[0].every_n_batches == 3


def test_add_plotting_callback(monkeypatch):
    # Add plotting callback
    import anemoi.training.diagnostics.callbacks.plot as plot

    class LossCurvePlot:
        def __init__(self, plotting_settings=None):
            pass

    monkeypatch.setattr(plot, "LossCurvePlot", LossCurvePlot)

    config = omegaconf.OmegaConf.create(yaml.safe_load(default_config))
    config.diagnostics.plot.callbacks = [{"_target_": "anemoi.training.diagnostics.callbacks.plot.LossCurvePlot"}]
    context = CallbacksContext(
        diagnostics=config.diagnostics,
        checkpoints_output=omegaconf.OmegaConf.create({"root": ".test_checkpoints"}),
        plots_output=None,
        wandb_enabled=False,
        mlflow_enabled=False,
    )
    callbacks = get_callbacks(context)
    assert len(callbacks) == NUM_FIXED_CALLBACKS + 1


def test_plot_loss_gathers_nan_mask_weights_from_nested_losses():
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

    callback = plot_mod.LossCurvePlot.__new__(plot_mod.LossCurvePlot)
    callback.every_n_batches = 1
    callback.dataset_names = ["data"]
    callback.parameter_groups = {}

    pl_module = MagicMock()
    pl_module.loss = {"data": combined_loss}
    pl_module.grid_dim = -2
    pl_module.grid_indices = {"data": MagicMock()}
    pl_module.grid_shard_sizes = {"data": None}
    pl_module.allgather_batch.side_effect = lambda tensor, *_args: tensor + 1.0

    # _prepare_batch is overridden in LossCurvePlot to snapshot and gather nan_mask_weights
    # before delegating batch preparation to the plot adapter. Call it directly.
    callback._prepare_batch(pl_module, batch={"data": torch.zeros((1, 1, 1, 3, 2))})

    assert pl_module.allgather_batch.call_count == 2
    for child_loss in callback.loss["data"].losses:
        torch.testing.assert_close(
            child_loss.loss.scaler.get_scaler_tensor("nan_mask_weights"),
            torch.full((1, 3, 2), 2.0),
        )


# Progress bar callback tests
progress_bar_config = """
training:
  method:
    _target_: anemoi.training.train.methods.EnsembleTraining

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

    callbacks = _get_progress_bar_callback(config.diagnostics)
    assert len(callbacks) == 0


def test_progress_bar_default():
    """Test that default TQDMProgressBar is used when progress_bar config has no _target_."""
    from pytorch_lightning.callbacks import TQDMProgressBar

    config = omegaconf.OmegaConf.create(yaml.safe_load(progress_bar_config))
    config.diagnostics.progress_bar = None  # No _target_ specified

    callbacks = _get_progress_bar_callback(config.diagnostics)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], TQDMProgressBar)


def test_progress_bar_custom():
    """Test that custom progress bar can be instantiated via _target_."""
    from pytorch_lightning.callbacks import RichProgressBar

    config = omegaconf.OmegaConf.create(yaml.safe_load(progress_bar_config))
    config.diagnostics.progress_bar = {
        "_target_": "pytorch_lightning.callbacks.RichProgressBar",
    }

    callbacks = _get_progress_bar_callback(config.diagnostics)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], RichProgressBar)
