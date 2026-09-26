# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for PerTimestepMetrics callback."""

import functools
from unittest.mock import MagicMock

import pytest
import torch

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.training.diagnostics.callbacks.per_timestep_metrics import PerTimestepMetrics
from anemoi.training.losses import MSELoss
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.step_output import TrainingStepOutput
from tests.batch_builders import build_batch

BS = 2
TIME = 6
ENS = 4
GRID = 16
NVAR = 3

_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)


@pytest.fixture
def callback() -> PerTimestepMetrics:
    return PerTimestepMetrics(every_n_batches=1)


@pytest.fixture
def callback_every_2() -> PerTimestepMetrics:
    return PerTimestepMetrics(every_n_batches=2)


def _gridded_batch(data: torch.Tensor, layout: TensorLayout = _LAYOUT) -> Batch:
    return build_batch(
        data={"data": data},
        coordinates={"data": torch.zeros(data.shape[layout.grid], 2)},
        layouts={"data": layout},
        variables={"data": [f"v{i}" for i in range(data.shape[layout.variables])]},
        static_coords={"data"},
    )


def _make_pl_module(
    n_timesteps: int = TIME,
    n_grid: int = GRID,
    n_var: int = NVAR,
) -> MagicMock:
    """Create a mocked pl_module with the attributes needed by the callback."""
    pl_module = MagicMock()

    # targets keep a single ensemble member; get_targets returns (targets, target_forcings)
    targets = _gridded_batch(torch.randn(BS, n_timesteps, 1, n_grid, n_var))
    pl_module.task.steps.return_value = ({"rollout_step": 0}, {"rollout_step": 1})
    pl_module.task.get_targets.return_value = (targets, None)
    pl_module.preprocess_targets.side_effect = lambda y: y

    # no grid sharding: return sources unchanged with a None slice, as the real method does.
    pl_module._prepare_tensors_for_loss.side_effect = lambda y_pred, y, **_: (y_pred, y, None)
    pl_module.logger_enabled = True

    # calculate_val_metrics returns a dict of metric_name -> tensor
    def mock_calculate_val_metrics(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "fkcrps_metric/data/pl": torch.tensor(1.0),
            "fkcrps_metric/data/sfc": torch.tensor(2.0),
        }

    pl_module.calculate_val_metrics = MagicMock(side_effect=mock_calculate_val_metrics)

    return pl_module


def _make_outputs(
    n_timesteps: int = TIME,
    n_ens: int = ENS,
    n_grid: int = GRID,
    n_var: int = NVAR,
) -> TrainingStepOutput:
    """Create outputs as returned by validation_step."""
    y_pred = _gridded_batch(torch.randn(BS, n_timesteps, n_ens, n_grid, n_var))
    return TrainingStepOutput(loss=torch.tensor(0.5), metrics={}, predictions=[y_pred])


def _make_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.precision = "32-true"
    return trainer


def _make_batch(n_timesteps: int = TIME) -> Batch:
    """Create a validation batch; the mocked task slices targets from it."""
    return _gridded_batch(torch.randn(BS, 2 + n_timesteps, 1, GRID, NVAR))


class TestPerTimestepMetrics:
    def test_init_default(self) -> None:
        cb = PerTimestepMetrics()
        assert cb.every_n_batches == 1

    def test_init_custom(self) -> None:
        cb = PerTimestepMetrics(every_n_batches=5)
        assert cb.every_n_batches == 5

    def test_skips_non_matching_batch(self, callback_every_2: PerTimestepMetrics) -> None:
        """Callback should skip batches that don't match every_n_batches."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        # batch_idx=1 should be skipped (1 % 2 != 0)
        callback_every_2.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=1)
        pl_module.calculate_val_metrics.assert_not_called()

    def test_runs_on_matching_batch(self, callback_every_2: PerTimestepMetrics) -> None:
        """Callback should run on batches matching every_n_batches."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        callback_every_2.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
        pl_module.calculate_val_metrics.assert_called()

    def test_calls_calculate_val_metrics_per_timestep(self, callback: PerTimestepMetrics) -> None:
        """Callback should call calculate_val_metrics once per timestep."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        # Should be called TIME times (once per timestep)
        assert pl_module.calculate_val_metrics.call_count == TIME

    def test_logs_per_timestep_metrics(self, callback: PerTimestepMetrics) -> None:
        """Callback should log metrics for each timestep and variable group."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        # Should have logged: TIME timesteps * 2 metric keys = 12 calls
        assert pl_module.log.call_count == TIME * 2

        # Check metric names include timestep suffix
        logged_names = [call.args[0] for call in pl_module.log.call_args_list]
        for t in range(1, TIME + 1):
            assert f"val_fkcrps_metric/data/pl/t_{t}" in logged_names
            assert f"val_fkcrps_metric/data/sfc/t_{t}" in logged_names

    def test_log_kwargs(self, callback: PerTimestepMetrics) -> None:
        """Check that log is called with correct kwargs."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        # Check first log call kwargs
        _, kwargs = pl_module.log.call_args_list[0]
        assert kwargs["on_epoch"] is True
        assert kwargs["on_step"] is False
        assert kwargs["prog_bar"] is False
        assert kwargs["sync_dist"] is True
        assert kwargs["batch_size"] == BS

    def test_handles_single_timestep(self, callback: PerTimestepMetrics) -> None:
        """Should work with a single output timestep."""
        trainer = _make_trainer()
        pl_module = _make_pl_module(n_timesteps=1)
        batch = _make_batch(n_timesteps=1)
        outputs = _make_outputs(n_timesteps=1)

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        # 1 timestep * 2 metric keys = 2 log calls
        assert pl_module.log.call_count == 2
        logged_names = [call.args[0] for call in pl_module.log.call_args_list]
        assert "val_fkcrps_metric/data/pl/t_1" in logged_names
        assert "val_fkcrps_metric/data/sfc/t_1" in logged_names

    def test_skips_when_no_outputs(self, callback: PerTimestepMetrics) -> None:
        """Should skip gracefully when outputs is empty or missing predictions."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()

        callback.on_validation_batch_end(trainer, pl_module, None, batch, batch_idx=0)
        pl_module.calculate_val_metrics.assert_not_called()

        empty_outputs = TrainingStepOutput(loss=torch.tensor(0.5), metrics={}, predictions=[])
        callback.on_validation_batch_end(trainer, pl_module, empty_outputs, batch, batch_idx=0)
        pl_module.calculate_val_metrics.assert_not_called()

    def test_slices_time_dimension_correctly(self, callback: PerTimestepMetrics) -> None:
        """Verify that calculate_val_metrics receives single-timestep slices."""
        trainer = _make_trainer()
        pl_module = _make_pl_module(n_timesteps=3)
        batch = _make_batch(n_timesteps=3)
        outputs = _make_outputs(n_timesteps=3)

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        # Check each call has time dim of size 1, and the prediction keeps its ensemble members
        for call in pl_module.calculate_val_metrics.call_args_list:
            y_pred_arg, y_arg = call.args
            assert y_pred_arg.time_size == 1
            assert y_arg.time_size == 1
            assert y_pred_arg.ensemble_size == ENS
            assert y_arg.ensemble_size == 1

    def test_passes_kwargs_to_calculate_val_metrics(self, callback: PerTimestepMetrics) -> None:
        """Verify kwargs passed to calculate_val_metrics."""
        trainer = _make_trainer()
        pl_module = _make_pl_module(n_timesteps=1)
        batch = _make_batch(n_timesteps=1)
        outputs = _make_outputs(n_timesteps=1)

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        _, kwargs = pl_module.calculate_val_metrics.call_args_list[0]
        assert kwargs["grid_shard_slice"] is None
        assert kwargs["dataset_name"] == "data"
        assert kwargs["without_scalers"] == ["time"]

    def test_scores_validation_outputs_without_rerunning_the_model(self, callback: PerTimestepMetrics) -> None:
        """The predictions come from validation_step; the targets are those of the first validation step."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)

        pl_module.assert_not_called()  # no second forward pass
        pl_module.task.steps.assert_called_once_with("validation")
        pl_module.task.get_targets.assert_called_once_with(
            batch,
            data_indices=pl_module.data_indices,
            rollout_step=0,
        )
        pl_module.preprocess_targets.assert_called_once()


def test_per_timestep_metrics_values_follow_the_layout_time_axis() -> None:
    """End to end through the real calculate_val_metrics, with the time axis in an unusual position."""
    layout = TensorLayout(batch=0, ensemble=1, grid=2, time=3, variables=4)
    # 2 members predicting 1 at t_1 and 2 at t_2 against zero targets with a single member
    pred_data = torch.ones(2, 2, 3, 2, 1)
    pred_data[:, :, :, 1, :] = 2.0
    target_data = torch.zeros(2, 1, 3, 2, 1)

    module = MagicMock()
    module.task.steps.return_value = ({},)
    module.task.get_targets.return_value = (_gridded_batch(target_data, layout), None)
    module.preprocess_targets.side_effect = lambda y: y
    module._prepare_tensors_for_loss.side_effect = lambda y_pred, y, **_: (y_pred, y, None)
    module._postprocess_dataset_view.side_effect = lambda view, _name, _layout: view
    module.metrics = {"data": {"mse": MSELoss()}}
    module.val_metric_ranges = {"data": {"all": [0]}}
    module.model_comm_group = None
    module.logger_enabled = True
    module.calculate_val_metrics = functools.partial(BaseTrainingModule.calculate_val_metrics, module)

    outputs = TrainingStepOutput(loss=torch.tensor(0.0), metrics={}, predictions=[_gridded_batch(pred_data, layout)])
    PerTimestepMetrics().on_validation_batch_end(_make_trainer(), module, outputs, _make_batch(), batch_idx=0)

    assert module.log.call_count == 2
    # MSE sums over the 3 grid points: 3 * 1^2 at t_1 and 3 * 2^2 at t_2
    for step, (call, expected) in enumerate(zip(module.log.call_args_list, [3.0, 12.0], strict=True), start=1):
        assert call.args[0] == f"val_mse_metric/data/all/t_{step}"
        torch.testing.assert_close(call.args[1], torch.tensor(expected))
