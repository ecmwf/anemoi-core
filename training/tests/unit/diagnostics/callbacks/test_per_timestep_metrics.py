# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for PerTimestepMetrics callback."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

from anemoi.training.diagnostics.callbacks.per_timestep_metrics import PerTimestepMetrics
from anemoi.training.losses.base import BaseLoss

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


class FakeLoss(BaseLoss):
    """Minimal BaseLoss subclass for testing."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: object) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor(1.0)

    @property
    def name(self) -> str:
        return "fake"


def _make_pl_module(
    n_timesteps: int = TIME,
    n_ens: int = ENS,
    n_grid: int = GRID,
    n_var: int = NVAR,
) -> MagicMock:
    """Create a mocked pl_module with the attributes needed by the callback."""
    pl_module = MagicMock()

    pred = torch.randn(BS, n_timesteps, n_ens, n_grid, n_var)
    target = torch.randn(BS, n_timesteps, n_grid, n_var)

    # task.get_inputs returns input dict
    pl_module.task.get_inputs.return_value = {"data": torch.randn(BS, 2, n_grid, n_var)}
    pl_module._expand_ens_dim.return_value = {"data": torch.randn(BS, 2, n_ens, n_grid, n_var)}

    # model forward returns predictions
    pl_module.__call__ = MagicMock(return_value={"data": pred})
    pl_module.return_value = {"data": pred}

    # task.get_targets returns targets
    y_full = {"data": target.unsqueeze(2)}  # add ens dim for _collapse_ens_dim
    pl_module.task.get_targets.return_value = y_full
    pl_module._collapse_ens_dim.return_value = {"data": target}

    # No ensemble comm group (single GPU case)
    pl_module.ens_comm_subgroup = None

    pl_module.model.post_processors = {"data": lambda x, **_: x}

    fake_loss = FakeLoss()
    pl_module.metrics = {"data": {"fkcrps": fake_loss}}

    pl_module.val_metric_ranges = {
        "data": {
            "pl": torch.arange(0, 2),
            "sfc": torch.arange(2, 3),
        },
    }

    pl_module.grid_shard_slice = {"data": None}
    pl_module.model_comm_group = None
    pl_module.logger_enabled = True
    pl_module.data_indices = MagicMock()

    return pl_module


def _make_trainer(precision: str = "32-true") -> MagicMock:
    trainer = MagicMock()
    trainer.precision = precision
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

        # batch_idx=1 should be skipped (1 % 2 != 0)
        callback_every_2.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=1)
        pl_module.task.get_inputs.assert_not_called()

    def test_runs_on_matching_batch(self, callback_every_2: PerTimestepMetrics) -> None:
        """Callback should run on batches matching every_n_batches."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()

        callback_every_2.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=0)
        pl_module.task.get_inputs.assert_called_once()

    def test_logs_per_timestep_metrics(self, callback: PerTimestepMetrics) -> None:
        """Callback should log metrics for each timestep and variable group."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()

        callback.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=0)

        # Should have logged: TIME timesteps * 2 var groups * 1 metric = 12 calls
        assert pl_module.log.call_count == TIME * 2

        # Check metric names
        logged_names = [call.args[0] for call in pl_module.log.call_args_list]
        for t in range(1, TIME + 1):
            assert f"val_fkcrps_metric/data/pl/t_{t}" in logged_names
            assert f"val_fkcrps_metric/data/sfc/t_{t}" in logged_names

    def test_log_kwargs(self, callback: PerTimestepMetrics) -> None:
        """Check that log is called with correct kwargs."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()

        callback.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=0)

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

        callback.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=0)

        # 1 timestep * 2 groups = 2 log calls
        assert pl_module.log.call_count == 2
        logged_names = [call.args[0] for call in pl_module.log.call_args_list]
        assert "val_fkcrps_metric/data/pl/t_1" in logged_names
        assert "val_fkcrps_metric/data/sfc/t_1" in logged_names

    def test_skips_non_baseloss_metrics(self, callback: PerTimestepMetrics) -> None:
        """Non-BaseLoss metrics should be skipped."""
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        batch = _make_batch()

        pl_module.metrics["data"]["non_loss"] = MagicMock(spec=[])

        callback.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=0)

        # Only BaseLoss metrics logged: TIME * 2 groups
        assert pl_module.log.call_count == TIME * 2

    def test_uses_autocast_for_mixed_precision(self, callback: PerTimestepMetrics) -> None:
        """Should apply autocast when precision is mixed."""
        trainer = _make_trainer(precision="16-mixed")
        pl_module = _make_pl_module()
        batch = _make_batch()

        # Should not raise
        callback.on_validation_batch_end(trainer, pl_module, [], batch, batch_idx=0)
        assert pl_module.log.call_count == TIME * 2

    def test_ensemble_gather(self, callback: PerTimestepMetrics) -> None:
        """Should call gather_tensor when ens_comm_subgroup is set."""
        pl_module = _make_pl_module()
        batch = _make_batch()

        pl_module.ens_comm_subgroup = MagicMock()
        pl_module.ens_comm_subgroup_size = 2

        with patch(
            "anemoi.models.distributed.graph.gather_tensor",
            side_effect=lambda x, **_: x,
        ):
            callback._eval_per_timestep(pl_module, batch)

        assert pl_module.log.call_count == TIME * 2
