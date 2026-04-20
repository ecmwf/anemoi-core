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
from anemoi.training.train.methods.diffusion import DiffusionTendencyTraining
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

    b, t, e, g, v = 2, 4, 1, 4, 2
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


# ── DATA_FULL target layout correctness ───────────────────────────────────────


def test_single_training_uses_data_full_target_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining._step must pass target_layout=DATA_FULL and the full (unfiltered) batch slice as y.

    When forcing variables are present the DATA_FULL variable count is larger than the
    DATA_OUTPUT count. Passing the already-filtered batch with DATA_FULL layout indices
    would cause an out-of-bounds CUDA assert.  This test verifies that get_targets()
    returns the full batch slice and that target_layout=DATA_FULL is declared.
    """
    # Include a forcing variable so DATA_FULL (3) > DATA_OUTPUT (2).
    name_to_index = {"prog_0": 0, "forcing_0": 1, "prog_1": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, forcing=["forcing_0"])}
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")

    forecaster = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(forecaster)
    _wire_training_module(forecaster, data_indices=data_indices, config=_CFG_EMPTY, task=task)
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
        ref = next(iter(y_pred.values()))
        return torch.zeros(1, dtype=ref.dtype, device=ref.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _compute_loss_metrics_stub)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)

    b, e, g = 2, 1, 4
    full_v = len(name_to_index)
    batch = {"data": torch.randn(b, 2, e, g, full_v)}
    forecaster._step(batch=batch, validation_mode=False)

    assert captured["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["target_layout"] == IndexSpace.DATA_FULL
    # Target must contain ALL variables (DATA_FULL), not just output vars (DATA_OUTPUT).
    assert captured["target_vars"] == full_v, (
        f"Expected target to have {full_v} (DATA_FULL) vars, got {captured['target_vars']}. "
        "get_targets() must not pre-filter to data.output.full."
    )


def test_ensemble_training_uses_data_full_target_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """EnsembleTraining._step must pass target_layout=DATA_FULL with the full batch slice as y."""
    name_to_index = {"prog_0": 0, "forcing_0": 1, "prog_1": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, forcing=["forcing_0"])}
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

    captured: dict[str, Any] = {}

    def _compute_loss_metrics_stub(
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        captured["target_vars"] = y["data"].shape[-1]
        captured["target_layout"] = kwargs["target_layout"]
        ref = next(iter(y_pred.values()))
        return torch.zeros(1, dtype=ref.dtype, device=ref.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _compute_loss_metrics_stub)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(task, "advance_input", lambda x, *_a, **_kw: x)

    b, e, g = 2, 1, 4
    full_v = len(name_to_index)
    batch = {"data": torch.randn(b, 2, e, g, full_v)}
    forecaster._step(batch=batch, validation_mode=False)

    assert captured["target_layout"] == IndexSpace.DATA_FULL
    assert (
        captured["target_vars"] == full_v
    ), f"Expected target to have {full_v} (DATA_FULL) vars, got {captured['target_vars']}."


def test_diffusion_training_uses_data_full_target_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """DiffusionTraining._step must pass target_layout=DATA_FULL with the full batch slice as y."""

    class _DummyDiffusionWrapper:
        def __init__(self, inner: DummyDiffusionModel) -> None:
            self.model = inner

    # Include an extra target (observation) variable to exercise the target-variable path.
    name_to_index = {"A": 0, "B": 1, "obs_A": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, target=["obs_A"])}
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
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *a, **kw: fn(*a, **kw))

    b, e, g = 2, 1, 4
    full_v = len(name_to_index)
    batch = {"data": torch.randn(b, 2, e, g, full_v)}
    forecaster._step(batch=batch, validation_mode=False)

    assert captured["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["target_layout"] == IndexSpace.DATA_FULL
    assert captured["pred_vars"] == len(data_indices["data"].model.output.full)
    assert (
        captured["target_vars"] == full_v
    ), f"Expected target to have {full_v} (DATA_FULL) vars, got {captured['target_vars']}."


def test_diffusion_training_target_reduction_fast_paths() -> None:
    """BaseDiffusionTraining.reduce_data_output_target_to_model_output uses fast paths when possible."""
    module = DiffusionTraining.__new__(DiffusionTraining)
    pl.LightningModule.__init__(module)

    # Identity: model.output == data.output (no target/forcing/diagnostic), so no selection needed.
    data_indices_identity = {"data": _make_minimal_index_collection({"A": 0, "B": 1})}
    _wire_training_module(module, data_indices=data_indices_identity, config=_CFG_DIFFUSION)
    y_identity = {"data": torch.randn((2, 1, 1, 4, 2), dtype=torch.float32)}
    y_model_identity = module.reduce_data_output_target_to_model_output(y_identity)
    assert y_model_identity["data"] is y_identity["data"]

    # Non-contiguous: obs_A sits between A and B in the original variable ordering
    # (name_to_index = {A:0, obs_A:1, B:2}) so data.output is [A, obs_A, B] (sorted by index).
    # model.output = [A, B] → positions [0, 2] in data.output, which is non-contiguous.
    data_indices_non_contiguous = {
        "data": _make_minimal_index_collection(
            {"A": 0, "obs_A": 1, "B": 2},
            target=["obs_A"],
        ),
    }
    _wire_training_module(module, data_indices=data_indices_non_contiguous, config=_CFG_DIFFUSION)
    y_non_contiguous = {"data": torch.randn((2, 1, 1, 4, 3), dtype=torch.float32)}
    y_model_non_contiguous = module.reduce_data_output_target_to_model_output(y_non_contiguous)
    expected_non_contiguous = y_non_contiguous["data"].index_select(
        -1,
        torch.tensor([0, 2], dtype=torch.long),
    )
    torch.testing.assert_close(y_model_non_contiguous["data"], expected_non_contiguous)
    assert (
        y_model_non_contiguous["data"].untyped_storage().data_ptr()
        != y_non_contiguous["data"].untyped_storage().data_ptr()
    )


def test_diffusion_tend_training_compute_dataset_loss_metrics_uses_data_full_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DiffusionTendencyTraining.compute_dataset_loss_metrics must use DATA_FULL for validation metrics.

    The state-space targets (y_state) originate from get_targets() which returns DATA_FULL.
    After applying the imputer inverse they are compared against the predicted state using
    target_layout=DATA_FULL so that LossVariableMapper can resolve variables correctly.
    """
    name_to_index = {"A": 0, "B": 1, "obs_A": 2}
    data_indices = {"data": _make_minimal_index_collection(name_to_index, target=["obs_A"])}

    forecaster = DiffusionTendencyTraining.__new__(DiffusionTendencyTraining)
    pl.LightningModule.__init__(forecaster)
    _wire_training_module(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION)

    captured: dict[str, Any] = {}

    class _DummyInner:
        def _apply_imputer_inverse(
            self,
            post_processors: dict[str, Any],
            dataset_name: str,
            x: torch.Tensor,
        ) -> torch.Tensor:
            del post_processors
            captured["inverse_dataset_name"] = dataset_name
            return x + 7.0

    class _DummyOuter:
        def __init__(self) -> None:
            self.model = _DummyInner()
            self.post_processors = {"data": object()}

    forecaster.model = _DummyOuter()

    def _prepare_tensors_stub(
        self: DiffusionTendencyTraining,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, slice]:
        del self, validation_mode, dataset_name
        return y_pred, y, slice(0, 1)

    def _compute_loss_stub(
        self: DiffusionTendencyTraining,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, y_pred, y
        captured["loss_kwargs"] = kwargs
        return torch.tensor(0.0)

    def _compute_metrics_stub(
        self: DiffusionTendencyTraining,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del self, y_pred
        captured["metric_target"] = y
        captured["metric_kwargs"] = kwargs
        return {"dummy": torch.tensor(1.0)}

    monkeypatch.setattr(DiffusionTendencyTraining, "_prepare_tensors_for_loss", _prepare_tensors_stub, raising=True)
    monkeypatch.setattr(DiffusionTendencyTraining, "_compute_loss", _compute_loss_stub, raising=True)
    monkeypatch.setattr(DiffusionTendencyTraining, "_compute_metrics", _compute_metrics_stub, raising=True)

    b, e, g = 2, 1, 4
    n_model = len(data_indices["data"].model.output.full)
    n_full = len(name_to_index)
    y_pred = torch.randn(b, 1, e, g, n_model, dtype=torch.float32)
    y = torch.randn(b, 1, e, g, len(data_indices["data"].data.output.full), dtype=torch.float32)
    y_pred_state = {"data": torch.randn(b, 1, e, g, n_model, dtype=torch.float32)}
    y_state = {"data": torch.randn(b, 1, e, g, n_full, dtype=torch.float32)}
    weights = {"data": torch.ones(b, 1, 1, 1, 1, dtype=torch.float32)}

    loss, metrics, _ = forecaster.compute_dataset_loss_metrics(
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
    # Loss kwargs keep the originally-passed layouts (DATA_OUTPUT for tendency targets)
    assert captured["loss_kwargs"]["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["loss_kwargs"]["target_layout"] == IndexSpace.DATA_OUTPUT
    # Metrics use DATA_FULL because state targets come from get_targets() in DATA_FULL space
    assert captured["metric_kwargs"]["pred_layout"] == IndexSpace.MODEL_OUTPUT
    assert captured["metric_kwargs"]["target_layout"] == IndexSpace.DATA_FULL
    # The metric target has been shifted by _apply_imputer_inverse (+7)
    torch.testing.assert_close(captured["metric_target"], y_state["data"] + 7.0)


def test_ensemble_compute_dataset_loss_metrics_forwards_data_full_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EnsembleTraining.compute_dataset_loss_metrics must forward target_layout=DATA_FULL."""
    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)

    forecaster.ens_comm_subgroup_size = 1
    forecaster.ens_comm_subgroup = None
    forecaster.grid_shard_slice = {"data": None}
    forecaster.grid_dim = -2
    forecaster.grid_shard_shapes = {"data": None}

    monkeypatch.setattr(
        "anemoi.training.train.methods.ensemble.gather_tensor",
        lambda input_, *_args, **_kwargs: input_,
        raising=True,
    )

    captured: dict[str, Any] = {}

    def _compute_loss_stub(
        self: EnsembleTraining,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, y_pred, y
        captured["loss_kwargs"] = kwargs
        return torch.tensor(0.0)

    def _compute_metrics_stub(
        self: EnsembleTraining,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del self, y_pred, y
        captured["metric_kwargs"] = kwargs
        return {"dummy": torch.tensor(1.0)}

    monkeypatch.setattr(EnsembleTraining, "_compute_loss", _compute_loss_stub, raising=True)
    monkeypatch.setattr(EnsembleTraining, "_compute_metrics", _compute_metrics_stub, raising=True)

    y_pred = torch.randn(2, 1, 2, 4, 3)
    y = torch.randn(2, 1, 4, 3)
    loss, metrics, y_pred_ens = forecaster.compute_dataset_loss_metrics(
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        rollout_step=3,
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
    assert captured["metric_kwargs"]["rollout_step"] == 3


def test_ensemble_make_targets_requires_singleton_ensemble_dim() -> None:
    """EnsembleTraining._make_targets must assert the ensemble dim is a singleton."""
    data_indices = _data_indices_single()
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")

    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)
    _wire_training_module(forecaster, data_indices=data_indices, config=_CFG_EMPTY, task=task)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1

    # Ensemble dim=2 (shape[2]=2) should trigger the singleton-ensemble assertion.
    # Use rollout_step=0 (start=0), requiring a batch with at least input+output=2 time steps.
    b, t, e, g, v = 2, 2, 2, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn((b, t, e, g, v), dtype=torch.float32)}

    with pytest.raises(AssertionError, match="Expected singleton ensemble dimension"):
        forecaster._collapse_ens_dim(batch)


# ── per-step metric key suffixes ──────────────────────────────────────────────


def test_single_compute_metrics_produces_per_step_key_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining._compute_metrics passes rollout_step as 'step' to calculate_val_metrics.

    Produces distinct metric key suffixes (/1, /2, ...) for each rollout step.
    """
    data_indices = _data_indices_single()
    task = Forecaster(multistep_input=1, multistep_output=1, timestep="6h")
    module = _make_single_training(task, data_indices)

    # Capture the 'step' kwarg that reaches calculate_val_metrics
    captured_steps: list[int | None] = []

    def _stub_calculate_val_metrics(
        _self: SingleTraining,
        _y_pred: torch.Tensor,
        _y: torch.Tensor,
        step: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        captured_steps.append(step)
        suffix = "" if step is None else f"/{step + 1}"
        return {f"rmse_metric/data/z500{suffix}": torch.tensor(1.0)}

    monkeypatch.setattr(SingleTraining, "calculate_val_metrics", _stub_calculate_val_metrics, raising=True)

    b, g, v = 2, 4, len(_NAME_TO_INDEX)
    y_pred = y = torch.zeros(b, g, v)

    metrics_step0 = module._compute_metrics(y_pred, y, dataset_name="data", rollout_step=0)
    metrics_step1 = module._compute_metrics(y_pred, y, dataset_name="data", rollout_step=1)

    assert captured_steps == [0, 1]
    assert "rmse_metric/data/z500/1" in metrics_step0
    assert "rmse_metric/data/z500/2" in metrics_step1
    # Keys must be distinct across steps
    assert set(metrics_step0.keys()).isdisjoint(set(metrics_step1.keys()))
