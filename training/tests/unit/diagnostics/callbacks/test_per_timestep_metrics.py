# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for PerTimestepMetrics callback."""

from unittest.mock import MagicMock

import pytest
import torch

from anemoi.models.data import TensorLayout
from anemoi.models.data.views import SourceView
from anemoi.models.data.views import create_source_view
from anemoi.training.diagnostics.callbacks.per_timestep_metrics import PerTimestepMetrics
from anemoi.training.losses import MSELoss
from anemoi.training.train.step_output import TrainingStepOutput

BS = 2
TIME = 6
ENS = 4
GRID = 16
NVAR = 3


@pytest.fixture
def callback() -> PerTimestepMetrics:
    return PerTimestepMetrics(every_n_batches=1)


@pytest.fixture
def callback_every_2() -> PerTimestepMetrics:
    return PerTimestepMetrics(every_n_batches=2)


def _make_pl_module(
    n_timesteps: int = TIME,
    n_grid: int = GRID,
    n_var: int = NVAR,
) -> MagicMock:
    """Create a mocked pl_module with the attributes needed by the callback."""
    pl_module = MagicMock()

    target = torch.randn(BS, n_timesteps, n_grid, n_var)

    # task.get_targets returns targets with ensemble dim
    y_full = {"data": target.unsqueeze(2)}
    pl_module.task.get_targets.return_value = y_full
    pl_module._collapse_ens_dim.return_value = {"data": target}

    pl_module.grid_shard_slice = {"data": None}
    # no grid sharding: return tensors unchanged with a None slice, as the real method does.
    pl_module._prepare_tensors_for_loss.side_effect = lambda y_pred, y, **_: (y_pred, y, None)
    pl_module.logger_enabled = True

    # calculate_val_metrics returns a dict of metric_name -> tensor
    def mock_calculate_val_metrics(
        _y_pred: torch.Tensor,
        _y: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
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
    val_loss = torch.tensor(0.5)
    metrics = {}
    y_pred = torch.randn(BS, n_timesteps, n_ens, n_grid, n_var)
    y_preds_dict = {"data": y_pred}
    return TrainingStepOutput(loss=val_loss, metrics=metrics, predictions=[y_preds_dict])


def _make_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.precision = "32-true"
    return trainer


def _make_batch(n_timesteps: int = TIME) -> dict[str, torch.Tensor]:
    """Create a batch dict with the expected structure."""
    total_steps = 2 + n_timesteps
    return {"data": torch.randn(BS, total_steps, GRID, NVAR)}


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
        """Should skip gracefully when outputs is empty or missing y_preds."""
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

        # Check each call has time dim of size 1
        for call in pl_module.calculate_val_metrics.call_args_list:
            y_pred_arg = call.args[0]
            y_arg = call.args[1]
            assert y_pred_arg.shape[1] == 1  # time dim
            assert y_arg.shape[1] == 1  # time dim

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

    def test_uses_collapse_ens_dim(self, callback: PerTimestepMetrics) -> None:
        """Should call _collapse_ens_dim on targets when available."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()
        outputs = _make_outputs()

        callback.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx=0)
        pl_module._collapse_ens_dim.assert_called_once()


def test_per_timestep_metrics_resolves_time_and_ensemble_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = TensorLayout(batch=0, ensemble=1, grid=2, time=3, variables=4)
    pred_data = torch.ones(2, 2, 3, 2, 1)
    pred_data[:, :, :, 1, :] = 2.0
    target_data = torch.zeros(2, 1, 3, 2, 1)

    def view(data: torch.Tensor) -> SourceView:
        return create_source_view(
            name="data",
            data=data,
            variables=["a"],
            statistics={},
            coordinates=torch.zeros(3, 2),
            layout=layout,
            coordinates_are_static=True,
        )

    pred, target = view(pred_data), view(target_data)
    module = MagicMock()
    module.task.get_inputs.return_value = {"data": pred}
    module.task.get_targets.return_value = ({"data": target}, None)
    module.preprocess_inputs.side_effect = lambda x: x
    module._expand_ens_dim.side_effect = lambda x: x
    module.preprocess_targets.side_effect = lambda x: x
    module.postprocess_targets.side_effect = lambda x: x
    module.return_value = {"data": pred}
    module.ens_comm_subgroup = object()
    module.ens_comm_subgroup_size = 2
    module.model.post_processors = {"data": lambda x, **_kwargs: x}
    module.metrics = {"data": {"mse": MSELoss()}}
    module.val_metric_ranges = {"data": {"all": [0]}}
    module._grid_shard_slice.return_value = None
    module.model_comm_group = None
    module.logger_enabled = True

    def gather(data: torch.Tensor, *, dim: int, sizes: list[int], mgroup: object) -> torch.Tensor:
        assert mgroup is module.ens_comm_subgroup
        assert dim == 1
        assert sizes == [2, 2]
        return torch.cat([data, data], dim=dim)

    monkeypatch.setattr("anemoi.models.distributed.graph.gather_tensor", gather)
    batch = MagicMock()
    batch.size = 2
    PerTimestepMetrics()._eval_per_timestep(module, batch)

    assert module.log.call_count == 2
    for step, (call, expected) in enumerate(zip(module.log.call_args_list, [3.0, 12.0], strict=True), start=1):
        assert call.args[0] == f"val_mse_metric/data/all/t_{step}"
        torch.testing.assert_close(call.args[1], torch.tensor(expected))
