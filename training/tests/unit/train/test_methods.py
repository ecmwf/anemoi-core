# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import einops
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.tasks import Autoencoder
from anemoi.training.tasks import Forecaster
from anemoi.training.tasks import TemporalDownscaler
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.diffusion import BaseDiffusionTraining
from anemoi.training.train.methods.diffusion import DiffusionTraining
from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.utils.index_space import IndexSpace
from anemoi.training.utils.masks import NoOutputMask


class DummyLoss(torch.nn.Module):
    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.mean((y_pred - y) ** 2)


class CaptureLoss(BaseLoss):
    """Loss that records every call for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.calls.append({"pred": pred, "target": target, "kwargs": kwargs})
        return torch.tensor(0.0)


class ShardingAwareCaptureLoss(CaptureLoss):
    @property
    def needs_shard_layout_info(self) -> bool:
        return True


class FakeGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class DummyModel:
    """Lightweight stub for AnemoiModelInterface that echoes tensor shapes."""

    def __init__(self, num_output_variables: int | None = None, output_times: int = 1) -> None:
        self.called_with: dict[str, Any] | None = None
        self.pre_processors = Processors([])
        self.post_processors = Processors([], inverse=True)
        self.output_times = output_times
        self.num_output_variables = num_output_variables
        self.metrics: dict = {}

    def _forward_tensor(
        self,
        x: torch.Tensor,
        model_comm_group: Any | None = None,
        grid_shard_slice: Any | None = None,
        grid_shard_shapes: Any | None = None,
    ) -> torch.Tensor:
        x_input = einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
        self.called_with = {
            "x_shape": tuple(x_input.shape),
            "model_comm_group": model_comm_group,
            "grid_shard_slice": grid_shard_slice,
            "grid_shard_shapes": grid_shard_shapes,
        }
        bs, _, e, g, v = x.shape
        output_vars = self.num_output_variables or v
        y_shape = (bs, self.output_times, e, g, output_vars)
        return torch.randn(y_shape, dtype=x.dtype, device=x.device)

    def __call__(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        model_comm_group: Any | None = None,
        grid_shard_slice: Any | None = None,
        grid_shard_shapes: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del kwargs
        if isinstance(x, dict):
            return {
                name: self._forward_tensor(
                    t,
                    model_comm_group=model_comm_group,
                    grid_shard_slice=grid_shard_slice,
                    grid_shard_shapes=grid_shard_shapes,
                )
                for name, t in x.items()
            }
        return self._forward_tensor(
            x,
            model_comm_group=model_comm_group,
            grid_shard_slice=grid_shard_slice,
            grid_shard_shapes=grid_shard_shapes,
        )


class DummyDiffusionModel(DummyModel):
    """Stub for a diffusion model wrapping DummyModel."""

    def __init__(self, num_output_variables: int | None = None) -> None:
        super().__init__(num_output_variables=num_output_variables, output_times=1)
        self.sigma_max = 4.0
        self.sigma_min = 1.0
        self.sigma_data = 0.5

    def fwd_with_preconditioning(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y_noised: torch.Tensor | dict[str, torch.Tensor],
        sigma: torch.Tensor | dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        pred = self(x, **kwargs)

        if isinstance(pred, dict):
            out: dict[str, torch.Tensor] = {}
            for dataset_name, pred_tensor in pred.items():
                sigma_tensor = sigma[dataset_name]
                y_noised_tensor = y_noised[dataset_name]
                assert sigma_tensor.shape[0] == pred_tensor.shape[0]
                assert all(sigma_tensor.shape[i] == 1 for i in range(1, sigma_tensor.ndim))
                if y_noised_tensor.ndim == 4:
                    y_noised_tensor = y_noised_tensor.unsqueeze(1)
                out[dataset_name] = y_noised_tensor + 0.1 * pred_tensor
            return out
        return pred


def _make_minimal_index_collection(
    name_to_index: dict[str, int],
    *,
    forcing: list[str] | None = None,
    diagnostic: list[str] | None = None,
    target: list[str] | None = None,
) -> IndexCollection:
    cfg = DictConfig(
        {
            "forcing": forcing or [],
            "diagnostic": diagnostic or [],
            "target": target or [],
        },
    )
    return IndexCollection(cfg, name_to_index)


_NAME_TO_INDEX: dict[str, int] = {"A": 0, "B": 1}


def _data_indices_single() -> dict[str, IndexCollection]:
    return {"data": _make_minimal_index_collection(_NAME_TO_INDEX)}


def _assert_step_return_format(
    loss: torch.Tensor,
    y_preds: list,
    expected_len: int,
    dataset_name: str = "data",
) -> None:
    """Assert the (loss, metrics, y_preds) contract of _step."""
    assert isinstance(loss, torch.Tensor)
    assert isinstance(y_preds, list)
    assert len(y_preds) == expected_len
    for pred in y_preds:
        assert isinstance(pred, dict)
        assert dataset_name in pred
        assert isinstance(pred[dataset_name], torch.Tensor)


def _wire_training_module(
    obj: BaseTrainingModule,
    *,
    data_indices: dict[str, IndexCollection],
    config: DictConfig,
    n_step_input: int = 1,
    n_step_output: int = 1,
    task: Any = None,
) -> None:
    """Wire the minimal set of attributes needed by ``__new__``-built test modules."""
    obj.data_indices = data_indices
    obj.dataset_names = list(data_indices.keys())
    obj.config = config
    obj.n_step_input = n_step_input
    obj.n_step_output = n_step_output
    obj.grid_dim = -2
    obj.model_comm_group = None
    obj.model_comm_group_size = 1
    obj.grid_shard_shapes = {"data": None}
    obj.grid_shard_slice = {"data": None}
    if task is not None:
        obj.task = task


# Shared minimal configs
_CFG_EMPTY = DictConfig({})
_CFG_DIFFUSION = DictConfig({"model": {"model": {"diffusion": {"rho": 7.0}}}})


# ── BaseTrainingModule: _compute_loss ──────────────────────────────────────────


def test_base_compute_loss_forwards_standard_loss_kwargs() -> None:
    """_compute_loss passes grid_shard_slice and model_comm_group to the loss function."""
    module = MagicMock(spec=BaseTrainingModule)
    loss = CaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.loss = {"data": loss}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": shard_shapes}

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)

    result = BaseTrainingModule._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        grid_shard_slice=grid_shard_slice,
        dataset_name="data",
    )

    assert torch.equal(result, torch.tensor(0.0))
    assert len(loss.calls) == 1
    assert loss.calls[0]["pred"] is y_pred
    assert loss.calls[0]["target"] is y
    assert loss.calls[0]["kwargs"] == {
        "grid_shard_slice": grid_shard_slice,
        "group": group,
    }


def test_base_compute_loss_forwards_sharding_metadata_when_requested() -> None:
    """_compute_loss adds grid_dim and grid_shard_shapes when loss.needs_shard_layout_info."""
    module = MagicMock(spec=BaseTrainingModule)
    loss = ShardingAwareCaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.loss = {"data": loss}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": shard_shapes}

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)

    result = BaseTrainingModule._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        grid_shard_slice=grid_shard_slice,
        dataset_name="data",
    )

    assert torch.equal(result, torch.tensor(0.0))
    assert loss.calls[0]["kwargs"] == {
        "grid_shard_slice": grid_shard_slice,
        "group": group,
        "grid_dim": -2,
        "grid_shard_shapes": shard_shapes,
    }


def test_base_compute_loss_forwards_shard_layout_to_combined_multiscale_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_loss correctly routes shard layout to a CombinedLoss wrapping MultiscaleLossWrapper."""
    module = MagicMock(spec=BaseTrainingModule)
    group = FakeGroup(size=2)
    grid_shard_shapes = [1, 1]
    pred = torch.randn(1, 1, 1, 2, 1)
    target = torch.randn(1, 1, 2, 1)
    grid_shard_slice = slice(0, 1)

    multiscale_loss = MultiscaleLossWrapper(
        per_scale_loss=MSELoss(),
        weights=[1.0],
        keep_batch_sharded=True,
    )
    prepare_for_smoothing = MagicMock(return_value=(pred, target, grid_shard_shapes, grid_shard_shapes))
    monkeypatch.setattr(multiscale_loss, "_prepare_for_smoothing", prepare_for_smoothing)
    monkeypatch.setattr("anemoi.training.losses.multiscale.gather_channels", lambda x, *_args: x)
    monkeypatch.setattr("anemoi.training.losses.base.reduce_tensor", lambda x, *_args: x)

    combined_loss = CombinedLoss(multiscale_loss)

    module.loss = {"data": combined_loss}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": grid_shard_shapes}

    result = BaseTrainingModule._compute_loss(
        module,
        y_pred=pred,
        y=target,
        grid_shard_slice=grid_shard_slice,
        dataset_name="data",
    )

    assert result.shape == (1,)
    prepare_for_smoothing.assert_called_once_with(pred, target, group, -2, grid_shard_shapes)


# ── BaseDiffusionTraining: _compute_loss ─────────────────────────────────────


def test_diffusion_compute_loss_forwards_standard_loss_kwargs() -> None:
    """DiffusionForecaster._compute_loss passes noise weights to the loss function."""
    module = MagicMock(spec=BaseDiffusionTraining)
    loss = CaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]
    weights = {"data": torch.tensor([0.25])}

    module.loss = {"data": loss}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": shard_shapes}

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)

    result = BaseDiffusionTraining._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        weights=weights,
        grid_shard_slice=grid_shard_slice,
    )

    assert torch.equal(result, torch.tensor(0.0))
    assert loss.calls[0]["kwargs"] == {
        "weights": weights["data"],
        "grid_shard_slice": grid_shard_slice,
        "group": group,
    }


def test_diffusion_compute_loss_forwards_sharding_metadata_when_requested() -> None:
    """DiffusionForecaster._compute_loss adds shard layout when loss.needs_shard_layout_info."""
    module = MagicMock(spec=BaseDiffusionTraining)
    loss = ShardingAwareCaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]
    weights = {"data": torch.tensor([0.25])}

    module.loss = {"data": loss}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": shard_shapes}

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)

    BaseDiffusionTraining._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        weights=weights,
        grid_shard_slice=grid_shard_slice,
    )

    assert loss.calls[0]["kwargs"] == {
        "weights": weights["data"],
        "grid_shard_slice": grid_shard_slice,
        "group": group,
        "grid_dim": -2,
        "grid_shard_shapes": shard_shapes,
    }


# ── BaseTrainingModule: calculate_val_metrics ──────────────────────────────────


def test_calculate_val_metrics_forwards_standard_metric_kwargs() -> None:
    """calculate_val_metrics passes scaler_indices, grid_shard_slice, group to each metric."""
    module = MagicMock(spec=BaseTrainingModule)
    metric = CaptureLoss()
    post_processor = MagicMock(side_effect=lambda x, **_: x)
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.model = MagicMock()
    module.model.post_processors = {"data": post_processor}
    module.metrics = {"data": {"multiscale": metric}}
    module.val_metric_ranges = {"data": {"z_500": [1]}}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": shard_shapes}

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)

    metrics = BaseTrainingModule.calculate_val_metrics(
        module,
        y_pred=y_pred,
        y=y,
        grid_shard_slice=grid_shard_slice,
        dataset_name="data",
        step=0,
    )

    assert "multiscale_metric/data/z_500/1" in metrics
    assert len(metric.calls) == 1
    assert metric.calls[0]["pred"] is y_pred
    assert metric.calls[0]["target"] is y
    assert metric.calls[0]["kwargs"] == {
        "scaler_indices": (..., [1]),
        "grid_shard_slice": grid_shard_slice,
        "group": group,
    }


def test_calculate_val_metrics_forwards_dataset_shard_shapes_when_requested() -> None:
    """calculate_val_metrics adds shard layout when metric.needs_shard_layout_info."""
    module = MagicMock(spec=BaseTrainingModule)
    metric = ShardingAwareCaptureLoss()
    post_processor = MagicMock(side_effect=lambda x, **_: x)
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.model = MagicMock()
    module.model.post_processors = {"data": post_processor}
    module.metrics = {"data": {"multiscale": metric}}
    module.val_metric_ranges = {"data": {"z_500": [1]}}
    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": shard_shapes}

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)

    metrics = BaseTrainingModule.calculate_val_metrics(
        module,
        y_pred=y_pred,
        y=y,
        grid_shard_slice=grid_shard_slice,
        dataset_name="data",
        step=0,
    )

    assert "multiscale_metric/data/z_500/1" in metrics
    assert metric.calls[0]["kwargs"] == {
        "scaler_indices": (..., [1]),
        "grid_shard_slice": grid_shard_slice,
        "group": group,
        "grid_dim": -2,
        "grid_shard_shapes": shard_shapes,
    }


# ── plot_adapter delegation ────────────────────────────────────────────────────


def test_training_module_plot_adapter_delegates_to_task() -> None:
    """BaseTrainingModule.plot_adapter is a transparent proxy to task._plot_adapter."""
    task = Autoencoder()
    module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(module)
    module.task = task
    assert module.plot_adapter is task._plot_adapter


def test_training_module_plot_adapter_reflects_forecaster_task() -> None:
    """When the task is a Forecaster, plot_adapter reports the correct output_times."""
    task = Forecaster(multistep_input=1, multistep_output=2, timestep="6h")
    module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(module)
    module.task = task
    assert module.plot_adapter is task._plot_adapter


# ── SingleTraining._step integration ──────────────────────────────────────────


def _make_single_training(task: Any, data_indices: dict[str, IndexCollection]) -> SingleTraining:
    """Build a SingleTraining module wired for unit tests."""
    module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(module)
    _wire_training_module(
        module,
        data_indices=data_indices,
        config=_CFG_EMPTY,
        n_step_input=task.num_input_timesteps,
        n_step_output=task.num_output_timesteps,
        task=task,
    )
    module.model = DummyModel(
        num_output_variables=len(next(iter(data_indices.values())).model.output),
        output_times=task.num_output_timesteps,
    )
    module.is_first_step = False
    module.updating_scalars = {}
    module.target_dataset_names = module.dataset_names
    module.loss = {"data": DummyLoss()}
    module.loss_supports_sharding = False
    module.metrics_support_sharding = True
    return module


def test_single_training_step_with_forecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining._step with Forecaster returns one y_pred per rollout step."""
    data_indices = _data_indices_single()
    # Rollout=1: steps = ({"rollout_step": 0},), offsets = [0h, +6h]
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    module = _make_single_training(task, data_indices)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(
        module,
        "compute_loss_metrics",
        lambda y_pred, _y, **_kw: (torch.tensor(0.0), {}, y_pred),
    )

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    # batch time-steps correspond to offsets [0h, +6h]
    batch = {"data": torch.randn(b, 2, e, g, v)}
    loss, _, y_preds = module._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)
    assert y_preds[0]["data"].shape == (b, 1, e, g, v)


def test_single_training_step_with_autoencoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining._step with Autoencoder returns one y_pred for the single step."""
    data_indices = _data_indices_single()
    task = Autoencoder()
    module = _make_single_training(task, data_indices)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(
        module,
        "compute_loss_metrics",
        lambda y_pred, _y, **_kw: (torch.tensor(0.0), {}, y_pred),
    )

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    # Autoencoder: single time step at t=0
    batch = {"data": torch.randn(b, 1, e, g, v)}
    loss, _, y_preds = module._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


def test_single_training_step_with_temporal_downscaler(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining._step with TemporalDownscaler returns one y_pred (single step)."""
    data_indices = _data_indices_single()
    # 18h → 6h without boundaries: interior offsets [6h, 12h], inputs [0h, 18h]
    task = TemporalDownscaler(
        input_timestep="18h",
        output_timestep="6h",
        output_left_boundary=False,
        output_right_boundary=False,
    )
    module = _make_single_training(task, data_indices)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(
        module,
        "compute_loss_metrics",
        lambda y_pred, _y, **_kw: (torch.tensor(0.0), {}, y_pred),
    )

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    # offsets = [0h, 6h, 12h, 18h] → 4 time steps
    batch = {"data": torch.randn(b, 4, e, g, v)}
    loss, _, y_preds = module._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


# ── SingleTraining: loss averaging ────────────────────────────────────────────


def test_single_training_loss_is_averaged_over_num_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """_step returns loss / num_steps (average over all rollout iterations)."""
    data_indices = _data_indices_single()
    # 2 steps at construction time so the loop runs twice
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
    )
    module = _make_single_training(task, data_indices)

    per_step_losses = iter([torch.tensor(2.0), torch.tensor(4.0)])
    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(1, 1, 1, 4, len(_NAME_TO_INDEX))}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "get_targets", lambda *_a, **_kw: dummy_y)
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)
    monkeypatch.setattr(module, "compute_loss_metrics", lambda *_a, **_kw: (next(per_step_losses), {}, dummy_y))

    b, e, g, v = 1, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 2, e, g, v)}
    loss, _, _ = module._step(batch, validation_mode=False)

    # Expected average: 3.0 = (2.0 + 4.0) / 2
    assert torch.isclose(loss, torch.tensor(3.0)), f"Expected 3.0, got {loss.item()}"


def test_single_training_advance_input_called_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """advance_input is invoked exactly once per rollout step."""
    data_indices = _data_indices_single()
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
    )
    module = _make_single_training(task, data_indices)

    advance_call_count: list[int] = []
    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(1, 1, 1, 4, len(_NAME_TO_INDEX))}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "get_targets", lambda *_a, **_kw: dummy_y)
    monkeypatch.setattr(module, "compute_loss_metrics", lambda *_a, **_kw: (torch.tensor(0.0), {}, dummy_y))
    monkeypatch.setattr(
        task,
        "advance_input",
        lambda x, *_a, **_kw: (advance_call_count.append(1), x)[1],
    )

    b, e, g, v = 1, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 2, e, g, v)}
    module._step(batch, validation_mode=False)

    assert len(advance_call_count) == task.num_steps


# ── DiffusionTraining._step integration ───────────────────────────────────────


def test_diffusion_training_step_with_forecaster() -> None:
    """DiffusionTraining._step returns a single (loss, metrics, [y_pred]) for a one-step forecaster."""

    class _DummyDiffusionWrapper:
        """Wraps DummyDiffusionModel to match model.model.fwd_with_preconditioning API."""

        def __init__(self, inner: DummyDiffusionModel) -> None:
            self.model = inner

    data_indices = _data_indices_single()
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")

    forecaster = DiffusionTraining.__new__(DiffusionTraining)
    pl.LightningModule.__init__(forecaster)
    _wire_training_module(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION, task=task)
    forecaster.model = _DummyDiffusionWrapper(
        DummyDiffusionModel(num_output_variables=len(next(iter(data_indices.values())).model.output)),
    )
    forecaster.rho = _CFG_DIFFUSION.model.model.diffusion.rho
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    # offsets=[0h, +6h] → 2 time steps
    batch = {"data": torch.randn(b, 2, e, g, v)}
    loss, _, y_preds = forecaster._step(batch={"data": batch["data"]}, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)
    assert y_preds[0]["data"].shape == (b, 1, e, g, v)


# ── EnsembleTraining: expand/collapse helpers ──────────────────────────────────


def test_ensemble_expand_ens_dim_tiles_ensemble_dimension() -> None:
    """_expand_ens_dim tiles the ensemble (dim=2) by nens_per_device."""
    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)
    forecaster.nens_per_device = 3

    b, t, e, g, v = 2, 1, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v)}
    expanded = forecaster._expand_ens_dim(batch)
    assert expanded["data"].shape == (b, t, 3, g, v)


def test_ensemble_collapse_ens_dim_takes_first_ensemble_member() -> None:
    """_collapse_ens_dim selects index 0 along the ensemble dimension."""
    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)

    b, t, e, g, v = 2, 1, 4, 4, 2
    sentinel = torch.full((b, t, g, v), 99.0)
    batch_tensor = torch.zeros(b, t, e, g, v)
    batch_tensor[:, :, 0, :, :] = sentinel
    batch = {"data": batch_tensor}

    collapsed = forecaster._collapse_ens_dim(batch)
    assert collapsed["data"].shape == (b, t, g, v)
    assert torch.all(collapsed["data"] == sentinel)


# ── EnsembleTraining._step integration ────────────────────────────────────────


def test_ensemble_training_step_with_forecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    """EnsembleTraining._step expands ensemble, runs forward, and collapses before loss."""
    data_indices = _data_indices_single()
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")

    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)
    _wire_training_module(forecaster, data_indices=data_indices, config=_CFG_EMPTY, task=task)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster.nens_per_device = 2
    forecaster.model = DummyModel(
        num_output_variables=len(next(iter(data_indices.values())).model.output),
        output_times=1,
    )
    forecaster.output_mask = {"data": NoOutputMask()}
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True

    def _stub_compute_loss_metrics(
        y_pred: dict[str, torch.Tensor],
        _y: dict[str, torch.Tensor],
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        ref = next(iter(y_pred.values()))
        return torch.zeros(1, dtype=ref.dtype, device=ref.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _stub_compute_loss_metrics)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))

    b, e_orig, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    # offsets=[0h, +6h] → 2 time steps; ensemble dim=1 (will be expanded to nens_per_device=2)
    batch = {"data": torch.randn(b, 2, e_orig, g, v)}
    loss, _, y_preds = forecaster._step(batch=batch, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(y_preds, list)
    assert len(y_preds) == task.num_steps  # 1 rollout step
    # y_pred shape: (b, n_step_output, nens_per_device, g, v)
    assert y_preds[0]["data"].shape == (b, 1, forecaster.nens_per_device, g, v)


# ── Multi-step rollout correctness ────────────────────────────────────────────


class _RecordingModel:
    """Wraps DummyModel and records a clone of every dict input it receives."""

    def __init__(self, inner: DummyModel) -> None:
        self._inner = inner
        self.recorded_x: list[dict[str, torch.Tensor]] = []

    def __call__(self, x: dict[str, torch.Tensor] | torch.Tensor, **kw: Any) -> Any:
        if isinstance(x, dict):
            self.recorded_x.append({k: v.clone() for k, v in x.items()})
        return self._inner(x, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_single_training_multi_rollout_accumulates_one_pred_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The y_preds list returned by _step has exactly one entry per rollout step."""
    data_indices = _data_indices_single()
    # offsets for rollout=3: [0h, 6h, 12h, 18h] → 4 time steps in batch
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 3, "maximum": 3},
        data_frequency="6h",
    )
    module = _make_single_training(task, data_indices)

    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(1, 1, 1, 4, len(_NAME_TO_INDEX))}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "get_targets", lambda *_a, **_kw: dummy_y)
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)
    monkeypatch.setattr(module, "compute_loss_metrics", lambda y_pred, _y, **_kw: (torch.tensor(0.0), {}, y_pred))

    b, e, g, v = 1, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 4, e, g, v)}
    _, _, y_preds = module._step(batch, validation_mode=False)

    assert len(y_preds) == 3, f"Expected 3 y_pred entries for rollout=3, got {len(y_preds)}"
    for pred in y_preds:
        assert pred["data"].shape == (b, 1, e, g, v)


def test_single_training_rollout_step_kwarg_propagated_to_get_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """task.get_targets is called with rollout_step=0 then rollout_step=1 in order.

    This ensures each rollout step fetches the correct target time slice from the batch,
    not the same slice repeated.
    """
    data_indices = _data_indices_single()
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
        data_frequency="6h",
    )
    module = _make_single_training(task, data_indices)

    captured_kwargs: list[dict] = []
    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(1, 1, 1, 4, len(_NAME_TO_INDEX))}

    def spy_get_targets(*_args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        captured_kwargs.append(kwargs.copy())
        return dummy_y

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "get_targets", spy_get_targets)
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)
    monkeypatch.setattr(module, "compute_loss_metrics", lambda y_pred, _y, **_kw: (torch.tensor(0.0), {}, y_pred))

    b, e, g, v = 1, 1, 4, len(_NAME_TO_INDEX)
    # offsets for rollout=2: [0h, +6h, +12h] → 3 time steps
    batch = {"data": torch.randn(b, 3, e, g, v)}
    module._step(batch, validation_mode=False)

    assert len(captured_kwargs) == 2, "get_targets must be called once per rollout step"
    assert captured_kwargs[0].get("rollout_step") == 0
    assert captured_kwargs[1].get("rollout_step") == 1


def test_single_training_rollout_step_kwarg_propagated_to_compute_loss_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_loss_metrics receives rollout_step=0 then rollout_step=1 in order.

    rollout_step is used inside compute_loss_metrics for per-step metric naming
    (e.g. _rstep0, _rstep1); wrong values would mis-name or collide metrics.
    """
    data_indices = _data_indices_single()
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
        data_frequency="6h",
    )
    module = _make_single_training(task, data_indices)

    captured_kwargs: list[dict] = []
    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(1, 1, 1, 4, len(_NAME_TO_INDEX))}

    def spy_compute_loss_metrics(
        y_pred: dict[str, torch.Tensor],
        _y: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        captured_kwargs.append(kwargs.copy())
        return torch.tensor(0.0), {}, y_pred

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "get_targets", lambda *_a, **_kw: dummy_y)
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)
    monkeypatch.setattr(module, "compute_loss_metrics", spy_compute_loss_metrics)

    b, e, g, v = 1, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 3, e, g, v)}
    module._step(batch, validation_mode=False)

    assert len(captured_kwargs) == 2, "compute_loss_metrics must be called once per rollout step"
    assert captured_kwargs[0].get("rollout_step") == 0
    assert captured_kwargs[1].get("rollout_step") == 1


def test_single_training_model_receives_updated_input_at_each_rollout_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second rollout step's forward pass receives the x returned by advance_input.

    This is the core plumbing test for the autoregressive loop: the output of
    ``advance_input`` at step N must become the model input at step N+1.
    We verify this by patching ``advance_input`` to add 1.0 to every element
    (a deterministic marker) and checking that recorded_x[1] == recorded_x[0] + 1.
    """
    data_indices = _data_indices_single()
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
        data_frequency="6h",
    )
    module = _make_single_training(task, data_indices)

    recorder = _RecordingModel(module.model)
    module.model = recorder

    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(1, 1, 1, 4, len(_NAME_TO_INDEX))}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "get_targets", lambda *_a, **_kw: dummy_y)
    # advance_input adds 1.0 so we can detect its result in the next input.
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: {k: v + 1.0 for k, v in x.items()})
    monkeypatch.setattr(module, "compute_loss_metrics", lambda y_pred, _y, **_kw: (torch.tensor(0.0), {}, y_pred))

    b, e, g, v = 1, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, 2, e, g, v)}
    module._step(batch, validation_mode=False)

    assert len(recorder.recorded_x) == 2, "Model should be called exactly once per rollout step"
    # The second input must be exactly (first input + 1.0) — i.e., the value that
    # advance_input returned — proving _step threads advance_input's output forward.
    assert torch.allclose(
        recorder.recorded_x[1]["data"],
        recorder.recorded_x[0]["data"] + 1.0,
    ), "Second forward pass must receive the x returned by advance_input, not the original"


def test_ensemble_training_multi_rollout_accumulates_one_pred_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnsembleTraining._step with rollout=2 appends one y_pred entry per rollout step."""
    data_indices = _data_indices_single()
    task = Forecaster(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 2, "maximum": 2},
        data_frequency="6h",
    )

    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)
    _wire_training_module(forecaster, data_indices=data_indices, config=_CFG_EMPTY, task=task)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster.nens_per_device = 2
    forecaster.model = DummyModel(
        num_output_variables=len(next(iter(data_indices.values())).model.output),
        output_times=1,
    )
    forecaster.output_mask = {"data": NoOutputMask()}
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True

    def _stub_compute_loss_metrics(
        y_pred: dict[str, torch.Tensor],
        _y: dict[str, torch.Tensor],
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        ref = next(iter(y_pred.values()))
        return torch.zeros(1, dtype=ref.dtype, device=ref.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _stub_compute_loss_metrics)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    dummy_y: dict[str, torch.Tensor] = {"data": torch.zeros(b, 1, 1, g, v)}
    monkeypatch.setattr(task, "get_targets", lambda *_a, **_kw: dummy_y)
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)

    batch = {"data": torch.randn(b, 2, e, g, v)}
    _, _, y_preds = forecaster._step(batch=batch, validation_mode=False)

    assert len(y_preds) == 2, f"Expected 2 y_pred entries for rollout=2, got {len(y_preds)}"
    for pred in y_preds:
        assert pred["data"].shape == (b, 1, forecaster.nens_per_device, g, v)


def test_graphmultioutinterpolator_uses_data_full_target_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-out interpolator should label targets as DATA_FULL."""
    forecaster = GraphMultiOutInterpolator.__new__(GraphMultiOutInterpolator)
    pl.LightningModule.__init__(forecaster)

    data_indices = _make_minimal_index_collection({"prog_0": 0, "forcing_0": 1, "prog_1": 2}, forcing=["forcing_0"])
    forecaster.data_indices = {"data": data_indices}
    forecaster.dataset_names = ["data"]
    forecaster.boundary_times = [0]
    forecaster.interp_times = [1, 2]
    forecaster.imap = {0: 0, 1: 1, 2: 2}
    forecaster.n_step_output = len(forecaster.interp_times)
    forecaster.n_step_input = 1
    forecaster.rollout = 1

    def _forward_stub(
        self: GraphMultiOutInterpolator,
        x_bound: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del self
        x = x_bound["data"]
        if x.ndim == 5:
            b, _, e, g, _ = x.shape
        else:
            b, e, g, _ = x.shape
        return {"data": torch.randn((b, 2, e, g, 2), dtype=x.dtype, device=x.device)}

    def _compute_loss_metrics_stub(
        self: GraphMultiOutInterpolator,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del self, y_pred
        assert kwargs["pred_layout"] == IndexSpace.MODEL_OUTPUT
        assert kwargs["target_layout"] == IndexSpace.DATA_FULL
        assert y["data"].shape[-1] == len(data_indices.name_to_index)
        return torch.tensor(0.0), {}, y

    monkeypatch.setattr(GraphMultiOutInterpolator, "forward", _forward_stub, raising=True)
    monkeypatch.setattr(GraphMultiOutInterpolator, "compute_loss_metrics", _compute_loss_metrics_stub, raising=True)

    b, e, g, v = 2, 1, 4, len(data_indices.name_to_index)
    batch = {"data": torch.randn((b, 3, e, g, v), dtype=torch.float32)}

    loss, metrics, y_preds = forecaster._step(batch=batch, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert isinstance(y_preds, list)
    assert len(y_preds) == 1
    assert y_preds[0]["data"].shape == (b, 2, e, g, 2)


def test_graphdiffusionforecaster_uses_reduced_model_target_and_full_loss_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyDiffusion:
        def __init__(self, model: DummyDiffusionModel) -> None:
            self.model = model

    name_to_index = {"A": 0, "B": 1, "obs_A": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, target=["obs_A"])}

    forecaster = GraphDiffusionForecaster.__new__(GraphDiffusionForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION)
    forecaster.model = DummyDiffusion(
        DummyDiffusionModel(num_output_variables=len(next(iter(data_indices.values())).model.output)),
    )
    forecaster.rho = _CFG_DIFFUSION.model.model.diffusion.rho
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True

    captured: dict[str, Any] = {}

    def _compute_loss_metrics_stub(
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        captured["pred_vars"] = y_pred["data"].shape[-1]
        captured["target_vars"] = y["data"].shape[-1]
        captured["pred_layout"] = kwargs["pred_layout"]
        captured["target_layout"] = kwargs["target_layout"]
        pred = y_pred["data"]
        return torch.zeros(1, dtype=pred.dtype, device=pred.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _compute_loss_metrics_stub)

    b, e, g, v = 2, 1, 4, len(name_to_index)
    t_in = _CFG_DIFFUSION.training.multistep_input
    t_out = _CFG_DIFFUSION.training.multistep_output
    batch = torch.randn((b, t_in + t_out, e, g, v), dtype=torch.float32)
    loss, _, _ = forecaster._step(batch={"data": batch}, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert captured["pred_vars"] == len(data_indices["data"].model.output.full)
    assert captured["target_vars"] == len(data_indices["data"].name_to_index)
    assert captured["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["target_layout"] == IndexSpace.DATA_FULL
    y_model, y_full = forecaster._get_diffusion_targets({"data": batch})
    y_data_output = forecaster.get_data_output_target(y_full)
    y_model_from_data_output = forecaster.reduce_data_output_target_to_model_output(y_data_output)
    assert y_model["data"].shape[-1] == len(data_indices["data"].model.output.full)
    assert y_data_output["data"].shape[-1] == len(data_indices["data"].data.output.full)
    torch.testing.assert_close(y_model["data"], y_model_from_data_output["data"])


def test_graphdiffusionforecaster_target_reduction_fast_paths() -> None:
    forecaster = GraphDiffusionForecaster.__new__(GraphDiffusionForecaster)
    pl.LightningModule.__init__(forecaster)

    data_indices_identity = {"data": _make_minimal_index_collection({"A": 0, "B": 1})}
    _set_base_task_attrs(forecaster, data_indices=data_indices_identity, config=_CFG_DIFFUSION)
    y_identity = {"data": torch.randn((2, 1, 1, 4, 2), dtype=torch.float32)}
    y_model_identity = forecaster.reduce_data_output_target_to_model_output(y_identity)
    assert y_model_identity["data"] is y_identity["data"]

    data_indices_non_contiguous = {
        "data": _make_minimal_index_collection(
            {"A": 0, "obs_A": 1, "B": 2},
            target=["obs_A"],
        ),
    }
    _set_base_task_attrs(forecaster, data_indices=data_indices_non_contiguous, config=_CFG_DIFFUSION)
    y_non_contiguous = {"data": torch.randn((2, 1, 1, 4, 3), dtype=torch.float32)}
    y_model_non_contiguous = forecaster.reduce_data_output_target_to_model_output(y_non_contiguous)
    expected_non_contiguous = y_non_contiguous["data"].index_select(
        -1,
        torch.tensor([0, 2], dtype=torch.long),
    )
    torch.testing.assert_close(y_model_non_contiguous["data"], expected_non_contiguous)
    assert (
        y_model_non_contiguous["data"].untyped_storage().data_ptr()
        != y_non_contiguous["data"].untyped_storage().data_ptr()
    )


def test_graphdiffusiontendforecaster_validation_uses_full_state_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTendencyDiffusionInner:
        sigma_max = 4.0
        sigma_min = 1.0
        sigma_data = 0.5

        def apply_reference_state_truncation(
            self,
            x: dict[str, torch.Tensor],
            _grid_shard_shapes: dict[str, Any],
            _model_comm_group: Any,
        ) -> dict[str, torch.Tensor]:
            return {dataset_name: x_tensor[:, -1:, :, :, :2].clone() for dataset_name, x_tensor in x.items()}

        def compute_tendency(
            self,
            x_t1: dict[str, torch.Tensor],
            x_t0: dict[str, torch.Tensor],
            _pre_processors_state: dict[str, Any],
            _pre_processors_tendencies: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            out = {}
            for dataset_name, x_t1_dataset in x_t1.items():
                tendency = x_t1_dataset.clone()
                tendency[..., :2] = x_t1_dataset[..., :2] - x_t0[dataset_name]
                out[dataset_name] = tendency
            return out

        def fwd_with_preconditioning(
            self,
            x: dict[str, torch.Tensor],
            y_noised: dict[str, torch.Tensor],
            sigma: dict[str, torch.Tensor],
            **_kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            del x, sigma
            out = {}
            for dataset_name, y_dataset in y_noised.items():
                assert y_dataset.shape[-1] == 2
                out[dataset_name] = torch.zeros_like(y_dataset[..., :2])
            return out

        def add_tendency_to_state(
            self,
            state_inp: dict[str, torch.Tensor],
            tendency: dict[str, torch.Tensor],
            _post_processors_state: dict[str, Any],
            _post_processors_tendencies: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            return {
                dataset_name: state_inp[dataset_name] + tendency_dataset
                for dataset_name, tendency_dataset in tendency.items()
            }

        def _apply_imputer_inverse(
            self,
            post_processors: dict[str, Any],
            dataset_name: str,
            x: torch.Tensor,
        ) -> torch.Tensor:
            del post_processors, dataset_name
            return x + 1.0

    class DummyTendencyDiffusion:
        def __init__(self) -> None:
            self.model = DummyTendencyDiffusionInner()
            self.pre_processors_tendencies = {"data": [object()]}
            self.pre_processors = {"data": object()}
            self.post_processors = {"data": object()}

    name_to_index = {"A": 0, "B": 1, "obs_A": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, target=["obs_A"])}

    forecaster = GraphDiffusionTendForecaster.__new__(GraphDiffusionTendForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION)
    forecaster.model = DummyTendencyDiffusion()
    forecaster.rho = _CFG_DIFFUSION.model.model.diffusion.rho
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True
    forecaster._tendency_pre_processors = {"data": [object()]}
    forecaster._tendency_post_processors = {"data": [object()]}

    captured: dict[str, Any] = {}

    def _compute_loss_metrics_stub(
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        captured["pred_vars"] = y_pred["data"].shape[-1]
        captured["target_vars"] = y["data"].shape[-1]
        captured["pred_layout"] = kwargs["pred_layout"]
        captured["target_layout"] = kwargs["target_layout"]
        captured["state_pred_vars"] = kwargs["y_pred_state"]["data"].shape[-1]
        captured["state_target_vars"] = kwargs["y_state"]["data"].shape[-1]
        return torch.zeros(1), {"dummy": torch.tensor(1.0)}, kwargs["y_pred_state"]

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _compute_loss_metrics_stub)

    b, e, g, v = 2, 1, 4, len(name_to_index)
    t_in = _CFG_DIFFUSION.training.multistep_input
    t_out = _CFG_DIFFUSION.training.multistep_output
    batch = {"data": torch.randn((b, t_in + t_out, e, g, v), dtype=torch.float32)}

    loss, metrics, y_preds = forecaster._step(batch=batch, validation_mode=True)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics["dummy"], torch.Tensor)
    assert captured["pred_vars"] == len(data_indices["data"].model.output.full)
    assert captured["target_vars"] == len(data_indices["data"].data.output.full)
    assert captured["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["target_layout"] == IndexSpace.DATA_OUTPUT
    assert captured["state_pred_vars"] == len(data_indices["data"].model.output.full)
    assert captured["state_target_vars"] == len(data_indices["data"].name_to_index)
    assert isinstance(y_preds, list)
    assert len(y_preds) == 1
    assert y_preds[0]["data"].shape[-1] == len(data_indices["data"].model.output.full)


def test_graphdiffusiontendforecaster_validation_metrics_use_data_full_state_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name_to_index = {"A": 0, "B": 1, "obs_A": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, target=["obs_A"])}

    forecaster = GraphDiffusionTendForecaster.__new__(GraphDiffusionTendForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION)

    captured: dict[str, Any] = {}

    class DummyInner:
        def _apply_imputer_inverse(
            self,
            post_processors: dict[str, Any],
            dataset_name: str,
            x: torch.Tensor,
        ) -> torch.Tensor:
            del post_processors
            captured["inverse_dataset_name"] = dataset_name
            return x + 7.0

    class DummyOuter:
        def __init__(self) -> None:
            self.model = DummyInner()
            self.post_processors = {"data": object()}

    forecaster.model = DummyOuter()

    def _prepare_tensors_for_loss_stub(
        self: GraphDiffusionTendForecaster,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, slice]:
        del self
        captured.setdefault("prepare_calls", []).append((validation_mode, dataset_name))
        return y_pred, y, slice(0, 1)

    def _compute_loss_stub(
        self: GraphDiffusionTendForecaster,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, y_pred, y
        captured["loss_kwargs"] = kwargs
        return torch.tensor(0.0)

    def _compute_metrics_stub(
        self: GraphDiffusionTendForecaster,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del self, y_pred
        captured["metric_target"] = y
        captured["metric_kwargs"] = kwargs
        return {"dummy": torch.tensor(1.0)}

    monkeypatch.setattr(
        GraphDiffusionTendForecaster,
        "_prepare_tensors_for_loss",
        _prepare_tensors_for_loss_stub,
        raising=True,
    )
    monkeypatch.setattr(GraphDiffusionTendForecaster, "_compute_loss", _compute_loss_stub, raising=True)
    monkeypatch.setattr(GraphDiffusionTendForecaster, "_compute_metrics", _compute_metrics_stub, raising=True)

    y_pred = torch.randn((2, 1, 1, 4, 2), dtype=torch.float32)
    y = torch.randn((2, 1, 1, 4, 3), dtype=torch.float32)
    y_pred_state = {"data": torch.randn((2, 1, 1, 4, 2), dtype=torch.float32)}
    y_state = {"data": torch.randn((2, 1, 1, 4, 3), dtype=torch.float32)}
    weights = {"data": torch.ones((2, 1, 1, 1, 1), dtype=torch.float32)}

    loss, metrics, y_pred_state_full = forecaster.compute_dataset_loss_metrics(
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        validation_mode=True,
        y_pred_state=y_pred_state,
        y_state=y_state,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_OUTPUT,
        weights=weights,
    )

    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics["dummy"], torch.Tensor)
    assert captured["inverse_dataset_name"] == "data"
    assert captured["loss_kwargs"]["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["loss_kwargs"]["target_layout"] == IndexSpace.DATA_OUTPUT
    assert captured["metric_kwargs"]["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["metric_kwargs"]["target_layout"] == IndexSpace.DATA_FULL
    torch.testing.assert_close(captured["metric_target"], y_state["data"] + 7.0)
    torch.testing.assert_close(y_pred_state_full, y_pred_state["data"])


def test_graphensforecaster_compute_dataset_loss_metrics_forwards_layout_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)

    forecaster.ens_comm_subgroup_size = 1
    forecaster.ens_comm_subgroup = None
    forecaster.grid_shard_slice = {"data": None}
    forecaster.grid_dim = -2
    forecaster.grid_shard_shapes = {"data": None}

    monkeypatch.setattr(
        "anemoi.training.train.tasks.ensforecaster.gather_tensor",
        lambda input_, *_args, **_kwargs: input_,
        raising=True,
    )

    captured: dict[str, dict[str, Any]] = {}

    def _compute_loss_stub(
        self: GraphEnsForecaster,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, y_pred, y
        captured["loss_kwargs"] = kwargs
        return torch.tensor(0.0)

    def _compute_metrics_stub(
        self: GraphEnsForecaster,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del self, y_pred, y
        captured["metric_kwargs"] = kwargs
        return {"dummy": torch.tensor(1.0)}

    monkeypatch.setattr(GraphEnsForecaster, "_compute_loss", _compute_loss_stub, raising=True)
    monkeypatch.setattr(GraphEnsForecaster, "_compute_metrics", _compute_metrics_stub, raising=True)

    y_pred = torch.randn(2, 1, 2, 4, 3)
    y = torch.randn(2, 1, 4, 3)
    loss, metrics, y_pred_ens = forecaster.compute_dataset_loss_metrics(
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        step=3,
        validation_mode=True,
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )

    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics["dummy"], torch.Tensor)
    assert y_pred_ens.shape == y_pred.shape
    assert captured["loss_kwargs"]["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["loss_kwargs"]["target_layout"] == IndexSpace.DATA_FULL
    assert captured["metric_kwargs"]["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["metric_kwargs"]["target_layout"] == IndexSpace.DATA_FULL
    assert captured["metric_kwargs"]["step"] == 3


def test_graphensforecaster_make_targets_requires_singleton_ensemble_dim() -> None:
    data_indices = _data_indices_single()
    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_FORECASTER, n_step_input=1, n_step_output=1)

    b, t, e, g, v = 2, 2, 2, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn((b, t, e, g, v), dtype=torch.float32)}

    with pytest.raises(AssertionError, match="Expected singleton ensemble dimension"):
        forecaster._make_targets(batch, start=1)
