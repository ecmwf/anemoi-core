from typing import Any
from unittest.mock import MagicMock

import einops
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
from anemoi.training.diagnostics.callbacks.plot_adapter import AutoencoderPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import DiffusionPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import InterpolatorMultiOutPlotAdapter
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.tasks import AutoencodingTask
from anemoi.training.tasks import ForecastingTask
from anemoi.training.tasks import TemporalDownscalingTask
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.train.methods.diffusion import BaseDiffusionForecaster
from anemoi.training.train.methods.diffusion import DiffusionTraining
from anemoi.training.train.methods.ensemble import EnsembleTraining
from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.utils.masks import NoOutputMask


class DummyLoss(torch.nn.Module):

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.mean((y_pred - y) ** 2)


class CaptureLoss(BaseLoss):
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
    def __init__(self, num_output_variables: int | None = None, output_times: int = 1, add_skip: bool = False) -> None:
        self.called_with: dict[str, Any] | None = None
        self.pre_processors = Processors([])
        self.post_processors = Processors([], inverse=True)
        self.output_times = output_times
        self.num_output_variables = num_output_variables
        self.add_skip = add_skip
        self.metrics = {}

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
    ) -> torch.Tensor:
        del kwargs
        if isinstance(x, dict):
            return {
                dataset_name: self._forward_tensor(
                    x_tensor,
                    model_comm_group=model_comm_group,
                    grid_shard_slice=grid_shard_slice,
                    grid_shard_shapes=grid_shard_shapes,
                )
                for dataset_name, x_tensor in x.items()
            }

        return self._forward_tensor(
            x,
            model_comm_group=model_comm_group,
            grid_shard_slice=grid_shard_slice,
            grid_shard_shapes=grid_shard_shapes,
        )


class DummyDiffusionModel(DummyModel):

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
        **kwargs,
    ) -> torch.Tensor:
        # behave like diffusion: call forward and combine
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

        assert sigma.shape[0] == x.shape[0]
        assert all(sigma.shape[i] == 1 for i in range(1, sigma.ndim))
        if y_noised.ndim == 4:
            y_noised = y_noised.unsqueeze(1)
        return y_noised + 0.1 * pred


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    cfg = DictConfig({"forcing": [], "diagnostic": [], "target": []})
    return IndexCollection(cfg, name_to_index)


# Shared test data: single-dataset name_to_index used in many tests.
_NAME_TO_INDEX = {"A": 0, "B": 1}


def _data_indices_single() -> dict[str, IndexCollection]:
    """Minimal data_indices for a single dataset 'data'."""
    return {"data": _make_minimal_index_collection(_NAME_TO_INDEX)}


def _assert_step_return_format(
    loss: torch.Tensor,
    y_preds: list,
    expected_len: int,
    dataset_name: str = "data",
) -> None:
    """Assert task _step return (loss, metrics, list of dicts) contract."""
    assert isinstance(loss, torch.Tensor)
    assert isinstance(y_preds, list)
    assert len(y_preds) == expected_len
    for pred in y_preds:
        assert isinstance(pred, dict)
        assert dataset_name in pred
        assert isinstance(pred[dataset_name], torch.Tensor)


#  Shared settings

_CFG_FORECASTER = DictConfig(
    {
        "training": {
            "multistep_input": 1,
            "multistep_output": 1,
            "rollout": {"start": 1, "epoch_increment": 1, "maximum": 3},
        },
    },
)


def _set_base_task_attrs(
    obj: BaseTrainingModule,
    *,
    data_indices: dict[str, IndexCollection],
    config: DictConfig,
    n_step_input: int = 1,
    n_step_output: int = 1,
    task: Any = None,
) -> None:
    """Set attributes common to tasks built via __new__ + pl.LightningModule.__init__."""
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


def test_graphforecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecaster output_times, get_init_step, and _step return shape (one instantiation)."""
    data_indices = _data_indices_single()

    # Build a SingleTraining module with a ForecastingTask
    training_module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(training_module)

    task = ForecastingTask(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={
            "start": _CFG_FORECASTER.training.rollout.start,
            "epoch_increment": _CFG_FORECASTER.training.rollout.epoch_increment,
            "maximum": _CFG_FORECASTER.training.rollout.maximum,
        },
    )
    _set_base_task_attrs(training_module, data_indices=data_indices, config=_CFG_FORECASTER, task=task)
    training_module.model = DummyModel(num_output_variables=len(next(iter(data_indices.values())).model.output))
    training_module.is_first_step = False
    training_module.updating_scalars = {}
    training_module.target_dataset_names = training_module.dataset_names
    training_module.loss = {"data": DummyLoss()}
    training_module.loss_supports_sharding = False
    training_module.metrics_support_sharding = True
    training_module._plot_adapter = ForecasterPlotAdapter(task)
    for i in range(task.rollout.maximum):
        task.rollout.step = i + 1
        assert training_module.plot_adapter.output_times == i + 1
        assert training_module.plot_adapter.get_init_step(i) == 0

    # _step returns one prediction per rollout step with shape (B, n_step_output, E, G, V)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(
        task,
        "advance_input",
        lambda x, *_args, **_kwargs: x,
    )

    required_time_steps = training_module.n_step_input + task.rollout.maximum * training_module.n_step_output
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, required_time_steps, e, g, v, dtype=torch.float32)}

    loss, _, y_preds = training_module._step(batch, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert len(y_preds) == 1
    for step_pred in y_preds:
        assert isinstance(step_pred, dict)
        assert "data" in step_pred
        pred = step_pred["data"]
        assert isinstance(pred, torch.Tensor)
        assert pred.ndim == 5
        assert pred.shape == (
            b,
            training_module.n_step_output,
            e,
            g,
            v,
        ), (
            f"Expected (B, n_step_output, E, G, V) = "
            f"({b}, {training_module.n_step_output}, {e}, {g}, {v}), got {pred.shape}"
        )


_CFG_DIFFUSION = DictConfig(
    {
        "training": {"multistep_input": 1, "multistep_output": 1},
        "model": {"model": {"diffusion": {"rho": 7.0}}},
    },
)


def test_graphdiffusionforecaster() -> None:
    class DummyDiffusion:
        def __init__(self, model: DummyDiffusionModel) -> None:
            self.model = model

    data_indices = _data_indices_single()
    task = ForecastingTask(
        multistep_input=_CFG_DIFFUSION.training.multistep_input,
        multistep_output=_CFG_DIFFUSION.training.multistep_output,
        timestep="6h",
    )
    forecaster = DiffusionTraining.__new__(DiffusionTraining)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION, task=task)
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

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    t = _CFG_DIFFUSION.training.multistep_input

    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)
    loss, _, y_preds = forecaster._step(batch={"data": batch}, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)
    y_pred = y_preds[0]["data"]
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, 1, e, g, v)


def test_base_compute_loss_forwards_standard_loss_kwargs() -> None:
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
    assert len(loss.calls) == 1
    assert loss.calls[0]["pred"] is y_pred
    assert loss.calls[0]["target"] is y
    assert loss.calls[0]["kwargs"] == {
        "grid_shard_slice": grid_shard_slice,
        "group": group,
        "grid_dim": -2,
        "grid_shard_shapes": shard_shapes,
    }


def test_base_compute_loss_forwards_shard_layout_to_combined_multiscale_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_diffusion_compute_loss_forwards_standard_loss_kwargs() -> None:
    module = MagicMock(spec=BaseDiffusionForecaster)
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

    result = BaseDiffusionForecaster._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        weights=weights,
        grid_shard_slice=grid_shard_slice,
    )

    assert torch.equal(result, torch.tensor(0.0))
    assert len(loss.calls) == 1
    assert loss.calls[0]["pred"] is y_pred
    assert loss.calls[0]["target"] is y
    assert loss.calls[0]["kwargs"] == {
        "weights": weights["data"],
        "grid_shard_slice": grid_shard_slice,
        "group": group,
    }


def test_diffusion_compute_loss_forwards_sharding_metadata_when_requested() -> None:
    module = MagicMock(spec=BaseDiffusionForecaster)
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

    result = BaseDiffusionForecaster._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_name="data",
        weights=weights,
        grid_shard_slice=grid_shard_slice,
    )

    assert torch.equal(result, torch.tensor(0.0))
    assert len(loss.calls) == 1
    assert loss.calls[0]["pred"] is y_pred
    assert loss.calls[0]["target"] is y
    assert loss.calls[0]["kwargs"] == {
        "weights": weights["data"],
        "grid_shard_slice": grid_shard_slice,
        "group": group,
        "grid_dim": -2,
        "grid_shard_shapes": shard_shapes,
    }


def test_calculate_val_metrics_forwards_standard_metric_kwargs() -> None:
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
    assert len(metric.calls) == 1
    assert metric.calls[0]["pred"] is y_pred
    assert metric.calls[0]["target"] is y
    assert metric.calls[0]["kwargs"] == {
        "scaler_indices": (..., [1]),
        "grid_shard_slice": grid_shard_slice,
        "group": group,
        "grid_dim": -2,
        "grid_shard_shapes": shard_shapes,
    }


def test_graphensforecaster_rollout_with_time_dim_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollout step works when model returns (B, T, E, G, V); _advance_input uses last time step."""
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)

    task = ForecastingTask(multistep_input=1, multistep_output=1, timestep="6h", rollout={"start": 1, "maximum": 1})

    forecaster = EnsembleTraining.__new__(EnsembleTraining)
    pl.LightningModule.__init__(forecaster)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster.nens_per_device = 2
    forecaster.model = DummyModel(num_output_variables=len(data_indices.model.output), output_times=1)
    forecaster.model_comm_group = None
    forecaster.model_comm_group_size = 1
    forecaster.grid_shard_shapes = {"data": None}
    forecaster.grid_shard_slice = {"data": None}
    forecaster.output_mask = {"data": NoOutputMask()}
    forecaster.data_indices = {"data": data_indices}
    forecaster.dataset_names = ["data"]
    forecaster.grid_dim = -2
    forecaster.task = task
    forecaster.target_dataset_names = ["data"]
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}

    def _compute_loss_metrics(
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        del y, args, kwargs
        pred = next(iter(y_pred.values()))
        return torch.zeros(1, dtype=pred.dtype, device=pred.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _compute_loss_metrics)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    b, g, v = 2, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn((b, forecaster.n_step_input + task.rollout.step, 1, g, v), dtype=torch.float32)}

    loss, _metrics, y_preds = forecaster._step(batch=batch, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(y_preds, list)
    assert len(y_preds) == task.rollout.step
    preds = y_preds[0]
    assert isinstance(preds, dict)
    assert preds["data"].ndim == 5
    assert preds["data"].shape == (b, 1, forecaster.nens_per_device, g, v)


@pytest.mark.parametrize(
    ("n_step_input", "n_step_output", "expected"),
    [
        (2, 3, [4.0, 5.0]),
        (2, 2, [3.0, 4.0]),
        (3, 2, [3.0, 4.0, 5.0]),
    ],
)
def test_rollout_advance_input_keeps_latest_steps(
    n_step_input: int,
    n_step_output: int,
    expected: list[float],
) -> None:
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)

    task = ForecastingTask(
        multistep_input=n_step_input,
        multistep_output=n_step_output,
        timestep="6h",
    )

    b, e, g, v = 1, 1, 2, len(_NAME_TO_INDEX)
    x = torch.zeros((b, n_step_input, e, g, v), dtype=torch.float32)
    for step in range(n_step_input):
        x[:, step] = float(step + 1)
    y_pred = torch.stack(
        [
            torch.full((b, e, g, v), float(n_step_input + step), dtype=torch.float32)
            for step in range(1, n_step_output + 1)
        ],
        dim=1,
    )
    batch = torch.zeros((b, n_step_input + n_step_output, e, g, v), dtype=torch.float32)

    updated = task._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        data_indices=data_indices,
    )
    kept_steps = updated[0, :, 0, 0, 0].tolist()
    expected_next_input = expected
    error_msg = (
        "Next input steps (used for the next forecast) "
        f"(n_step_input={n_step_input}, n_step_output={n_step_output}) "
        f"should be {expected_next_input}, got {kept_steps}."
    )
    assert kept_steps == expected_next_input, error_msg
    for idx, value in enumerate(expected):
        assert torch.all(updated[:, idx] == value)


# Minimal index stub for interpolator output_times tests (no full IndexCollection).
class _DummyIndexForInterpolator:
    model = type("_Dummy", (), {"output": [0]})()


_CFG_INTERP_TWO_TARGETS = DictConfig(
    {
        "training": {
            "explicit_times": {
                "input": ["2025-01-01T00"],
                "target": ["2025-01-01T00", "2025-01-01T06"],
            },
        },
    },
)

# Config for interpolator _step tests (numeric indices): 2 boundary, 2 target steps.
_CFG_INTERP_STEP = DictConfig({"training": {"explicit_times": {"input": [0, 3], "target": [1, 2]}}})

# Autoencoder config
_CFG_AE = DictConfig({"training": {"multistep_input": 1, "multistep_output": 1}})


class _InterpolatorStub:
    """Minimal stub with attributes needed by InterpolatorMultiOutPlotAdapter."""

    def __init__(self, n_step_input: int, n_step_output: int, interp_times: list) -> None:
        self.n_step_input = n_step_input
        self.n_step_output = n_step_output
        self.interp_times = interp_times
        self._plot_adapter = None

    @property
    def plot_adapter(self) -> None:
        return self._plot_adapter


def test_interpolator_output_times_and_get_init_step() -> None:
    """Interpolator task: output_times == len(target), get_init_step(i) == i."""
    interp_times = _CFG_INTERP_TWO_TARGETS.training.explicit_times.target
    stub = _InterpolatorStub(
        n_step_input=1,
        n_step_output=len(interp_times),
        interp_times=interp_times,
    )
    stub._plot_adapter = InterpolatorMultiOutPlotAdapter(stub)

    assert stub.plot_adapter.output_times == 2
    for i in range(stub.plot_adapter.output_times):
        assert stub.plot_adapter.get_init_step(i) == i


# ---- output_times / get_init_step / _step return format for all tasks ----


def test_graphdiffusionforecaster_output_times_and_get_init_step() -> None:
    """Diffusion has output_times=1 and uses base get_init_step (returns 0)."""
    forecaster = DiffusionTraining.__new__(DiffusionTraining)
    pl.LightningModule.__init__(forecaster)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster._plot_adapter = DiffusionPlotAdapter(forecaster)
    assert forecaster.plot_adapter.output_times == 1
    assert forecaster.plot_adapter.get_init_step(0) == 0


def test_graphforecaster_get_init_step() -> None:
    """Forecaster get_init_step(rollout_step) returns 0 for all steps."""
    task = ForecastingTask(
        multistep_input=1,
        multistep_output=1,
        timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
    )
    adapter = ForecasterPlotAdapter(task)
    assert adapter.output_times == 1
    assert adapter.get_init_step(0) == 0
    assert adapter.get_init_step(1) == 0


def test_graphautoencoder_output_times() -> None:
    """Autoencoder training module has output_times=1."""
    data_indices = _data_indices_single()
    ae = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(ae)
    _set_base_task_attrs(ae, data_indices=data_indices, config=_CFG_AE)
    ae._plot_adapter = AutoencoderPlotAdapter(ae)
    assert ae.plot_adapter.output_times == 1


def test_graphautoencoder_step_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining + AutoencodingTask _step returns (loss, metrics, [y_pred]) for consistent task contract."""

    def dummy_forward(x: dict) -> dict:
        b = next(iter(x.values())).shape[0]
        t = next(iter(x.values())).shape[1]
        e = next(iter(x.values())).shape[2]
        g = next(iter(x.values())).shape[3]
        v = next(iter(x.values())).shape[4]
        return {dn: torch.randn(b, t, e, g, v, dtype=torch.float32) for dn in x}

    data_indices = _data_indices_single()
    task = AutoencodingTask()
    ae = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(ae)
    _set_base_task_attrs(ae, data_indices=data_indices, config=_CFG_AE, task=task)
    ae.model = type(
        "M",
        (),
        {"__call__": lambda _self, x, **_kwargs: dummy_forward(x)},
    )()

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(
        ae,
        "compute_loss_metrics",
        lambda *args, **_kwargs: (torch.tensor(0.0), {}, args[0] if args else None),
    )

    b, t, e, g, v = 2, 1, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, _, y_preds = ae._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


def test_temporal_downscaling_step_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """SingleTraining + TemporalDownscalingTask _step returns (loss, metrics, [y_pred]) for plot callback contract."""

    def dummy_forward(x_bound: dict) -> dict:
        b = next(iter(x_bound.values())).shape[0]
        e = next(iter(x_bound.values())).shape[2]
        g = next(iter(x_bound.values())).shape[3]
        v = next(iter(x_bound.values())).shape[4]
        return {"data": torch.randn(b, 2, e, g, v, dtype=torch.float32)}

    data_indices = _data_indices_single()
    task = TemporalDownscalingTask(
        input_timestep="18h",
        output_timestep="6H",
        output_left_boundary=False,
        output_right_boundary=False,
    )
    training_module = SingleTraining.__new__(SingleTraining)
    pl.LightningModule.__init__(training_module)
    _set_base_task_attrs(
        training_module,
        data_indices=data_indices,
        config=_CFG_INTERP_STEP,
        n_step_output=task.num_output_timesteps,
        n_step_input=task.num_input_timesteps,
        task=task,
    )
    training_module.model = type(
        "M",
        (),
        {"__call__": lambda _self, x, **_kwargs: dummy_forward(x)},
    )()
    training_module.loss = {"data": DummyLoss()}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(
        training_module,
        "compute_loss_metrics",
        lambda *args, **_kwargs: (torch.tensor(0.0), {}, args[0] if args else None),
    )

    b, t, e, g, v = 2, 4, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, _, y_preds = training_module._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)
