from typing import Any
from unittest.mock import MagicMock

import einops
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.diagnostics.callbacks.plot_adapter import AutoencoderPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import DiffusionPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import ForecasterPlotAdapter
from anemoi.training.diagnostics.callbacks.plot_adapter import InterpolatorMultiOutPlotAdapter
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.multiscale import MultiscaleLossWrapper
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.train.tasks.diffusionforecaster import GraphDiffusionForecaster
from anemoi.training.train.tasks.diffusionforecaster import GraphDiffusionTendForecaster
from anemoi.training.train.tasks.ensforecaster import GraphEnsForecaster
from anemoi.training.train.tasks.forecaster import GraphForecaster
from anemoi.training.train.tasks.interpolator import GraphMultiOutInterpolator
from anemoi.training.utils.dataset_context import DatasetContext
from anemoi.training.utils.dataset_context import DatasetContextDynamic
from anemoi.training.utils.dataset_context import DatasetContextStatic
from anemoi.training.utils.masks import NoOutputMask

_RAW_DATA_INDICES_LOOKUP_ERROR = "raw data_indices lookup should not be used here"


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
    def __init__(
        self,
        num_output_variables: int | None = None,
        output_times: int = 1,
        add_skip: bool = False,
    ) -> None:
        self.called_with: dict[str, Any] | None = None
        self.pre_processors = {"data": lambda x: x}
        self.post_processors = {"data": lambda x, **_kwargs: x}
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


class _ExplodingDataIndices:
    @property
    def data(self) -> Any:
        raise AssertionError(_RAW_DATA_INDICES_LOOKUP_ERROR)


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
            "rollout": {"start": 1, "epoch_increment": 1, "max": 3},
        },
    },
)


def _set_base_task_attrs(
    obj: BaseGraphModule,
    *,
    data_indices: dict[str, IndexCollection],
    config: DictConfig,
    n_step_input: int = 1,
    n_step_output: int = 1,
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
    obj.output_mask = {"data": NoOutputMask()}
    obj.loss = {"data": DummyLoss()}
    obj.metrics = {"data": {}}
    obj.val_metric_ranges = {"data": {}}


def _make_dataset_context(
    *,
    name: str = "data",
    loss: BaseLoss | torch.nn.Module | None = None,
    metrics: dict[str, Any] | None = None,
    val_metric_ranges: dict[str, Any] | None = None,
    post_processor: Any | None = None,
    effective_grid_shard_slice: slice | None = None,
    grid_shard_shapes: Any = None,
) -> DatasetContext:
    return DatasetContext(
        static=DatasetContextStatic(
            name=name,
            loss=loss,
            metrics={} if metrics is None else metrics,
            val_metric_ranges={} if val_metric_ranges is None else val_metric_ranges,
            pre_processor=torch.nn.Identity(),
            pre_processor_tendencies=None,
            post_processor=post_processor if post_processor is not None else MagicMock(side_effect=lambda x, **_: x),
            post_processor_tendencies=None,
            data_indices=_make_minimal_index_collection(_NAME_TO_INDEX),
            output_mask=NoOutputMask(),
        ),
        dynamic=DatasetContextDynamic(
            batch_grid_shard_slice=effective_grid_shard_slice,
            effective_grid_shard_slice=effective_grid_shard_slice,
            grid_shard_shapes=grid_shard_shapes,
        ),
    )


def test_refresh_dataset_context_static_supports_non_target_datasets() -> None:
    """Datasets without a training loss should still get a static context."""
    data_indices = {
        "data": _make_minimal_index_collection(_NAME_TO_INDEX),
        "aux": _make_minimal_index_collection(_NAME_TO_INDEX),
    }
    forecaster = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.dataset_names = list(data_indices.keys())
    forecaster.data_indices = data_indices
    forecaster.output_mask = {"data": NoOutputMask(), "aux": NoOutputMask()}
    forecaster.loss = {"data": DummyLoss()}  # "aux" intentionally excluded from target datasets
    forecaster.metrics = {"data": {}}
    forecaster.val_metric_ranges = {"data": {}}

    model = DummyModel(num_output_variables=len(next(iter(data_indices.values())).model.output))
    model.pre_processors = {"data": lambda x: x, "aux": lambda x: x}
    model.post_processors = {
        "data": lambda x, **_kwargs: x,
        "aux": lambda x, **_kwargs: x,
    }
    forecaster.model = model

    forecaster.refresh_dataset_context_static()

    assert set(forecaster.dataset_context_static.keys()) == {"data", "aux"}
    assert forecaster.dataset_context_static["data"].loss is not None
    assert forecaster.dataset_context_static["aux"].loss is None
    assert forecaster.dataset_context_static["aux"].metrics == {}
    assert forecaster.dataset_context_static["aux"].val_metric_ranges == {}


def test_graphforecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecaster output_times, get_init_step, and _step return shape (one instantiation)."""
    data_indices = _data_indices_single()
    forecaster = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_FORECASTER)
    forecaster.rollout = _CFG_FORECASTER.training.rollout.start
    forecaster.rollout_epoch_increment = _CFG_FORECASTER.training.rollout.epoch_increment
    forecaster.rollout_max = _CFG_FORECASTER.training.rollout.max
    forecaster.model = DummyModel(num_output_variables=len(next(iter(data_indices.values())).model.output))
    forecaster.is_first_step = False
    forecaster.updating_scalars = {}
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.loss = {"data": DummyLoss()}
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True
    forecaster.refresh_dataset_context_static()
    forecaster._plot_adapter = ForecasterPlotAdapter(forecaster)

    assert forecaster.plot_adapter.output_times == 1
    for i in range(1, _CFG_FORECASTER.training.rollout.max + 1):
        forecaster.rollout = i
        assert forecaster.plot_adapter.get_init_step(i) == 0
        assert forecaster.plot_adapter.output_times == i

    # _step returns one prediction per rollout step with shape (B, n_step_output, E, G, V)
    monkeypatch.setattr(
        "torch.utils.checkpoint.checkpoint",
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    monkeypatch.setattr(
        forecaster,
        "_advance_input",
        lambda x, *_args, **_kwargs: x,
    )

    forecaster.rollout = 2
    required_time_steps = forecaster.n_step_input + forecaster.rollout * forecaster.n_step_output
    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    batch = {"data": torch.randn(b, required_time_steps, e, g, v, dtype=torch.float32)}

    loss, _, y_preds = forecaster._step(batch, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert len(y_preds) == forecaster.rollout
    for step_pred in y_preds:
        assert isinstance(step_pred, dict)
        assert "data" in step_pred
        pred = step_pred["data"]
        assert isinstance(pred, torch.Tensor)
        assert pred.ndim == 5
        assert pred.shape == (
            b,
            forecaster.n_step_output,
            e,
            g,
            v,
        ), f"Expected (B, n_step_output, E, G, V) = ({b}, {forecaster.n_step_output}, {e}, {g}, {v}), got {pred.shape}"


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
            self.pre_processors = model.pre_processors
            self.post_processors = model.post_processors

    data_indices = _data_indices_single()
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
    forecaster.refresh_dataset_context_static()

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    t = _CFG_DIFFUSION.training.multistep_input

    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)
    loss, _, y_preds = forecaster._step(batch={"data": batch}, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)
    y_pred = y_preds[0]["data"]
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, 1, e, g, v)


def test_base_compute_loss_forwards_standard_loss_kwargs() -> None:
    module = MagicMock(spec=BaseGraphModule)
    loss = CaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)
    dataset_ctx = _make_dataset_context(
        loss=loss,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=shard_shapes,
    )

    result = BaseGraphModule._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_ctx=dataset_ctx,
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
    module = MagicMock(spec=BaseGraphModule)
    loss = ShardingAwareCaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)
    dataset_ctx = _make_dataset_context(
        loss=loss,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=shard_shapes,
    )

    result = BaseGraphModule._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_ctx=dataset_ctx,
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
    module = MagicMock(spec=BaseGraphModule)
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

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2
    dataset_ctx = _make_dataset_context(
        loss=combined_loss,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=grid_shard_shapes,
    )

    result = BaseGraphModule._compute_loss(
        module,
        y_pred=pred,
        y=target,
        dataset_ctx=dataset_ctx,
    )

    assert result.shape == (1,)
    prepare_for_smoothing.assert_called_once_with(pred, target, group, -2, grid_shard_shapes)


def test_diffusion_compute_loss_forwards_standard_loss_kwargs() -> None:
    module = MagicMock(spec=GraphDiffusionForecaster)
    loss = CaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]
    weights = {"data": torch.tensor([0.25])}

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)
    dataset_ctx = _make_dataset_context(
        loss=loss,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=shard_shapes,
    )

    result = GraphDiffusionForecaster._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_ctx=dataset_ctx,
        weights=weights,
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
    module = MagicMock(spec=GraphDiffusionForecaster)
    loss = ShardingAwareCaptureLoss()
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]
    weights = {"data": torch.tensor([0.25])}

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)
    dataset_ctx = _make_dataset_context(
        loss=loss,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=shard_shapes,
    )

    result = GraphDiffusionForecaster._compute_loss(
        module,
        y_pred=y_pred,
        y=y,
        dataset_ctx=dataset_ctx,
        weights=weights,
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
    module = MagicMock(spec=BaseGraphModule)
    metric = CaptureLoss()
    post_processor = MagicMock(side_effect=lambda x, **_: x)
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)
    dataset_ctx = _make_dataset_context(
        metrics={"multiscale": metric},
        val_metric_ranges={"z_500": [1]},
        post_processor=post_processor,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=shard_shapes,
    )

    metrics = BaseGraphModule.calculate_val_metrics(
        module,
        y_pred=y_pred,
        y=y,
        dataset_ctx=dataset_ctx,
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
    module = MagicMock(spec=BaseGraphModule)
    metric = ShardingAwareCaptureLoss()
    post_processor = MagicMock(side_effect=lambda x, **_: x)
    group = object()
    shard_shapes = [(1, 1, 1, 2, 3), (1, 1, 1, 2, 3)]

    module.model_comm_group = group
    module.model_comm_group_size = 2
    module.grid_dim = -2

    y_pred = torch.randn(1, 1, 1, 2, 3)
    y = torch.randn(1, 1, 2, 3)
    grid_shard_slice = slice(0, 2)
    dataset_ctx = _make_dataset_context(
        metrics={"multiscale": metric},
        val_metric_ranges={"z_500": [1]},
        post_processor=post_processor,
        effective_grid_shard_slice=grid_shard_slice,
        grid_shard_shapes=shard_shapes,
    )

    metrics = BaseGraphModule.calculate_val_metrics(
        module,
        y_pred=y_pred,
        y=y,
        dataset_ctx=dataset_ctx,
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

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices={"data": data_indices}, config=_CFG_FORECASTER)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster.rollout = 1
    forecaster.nens_per_device = 2
    forecaster.model = DummyModel(num_output_variables=len(data_indices.model.output), output_times=1)
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.refresh_dataset_context_static()

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
    b, g, v = 2, 4, len(_NAME_TO_INDEX)
    batch = {
        "data": torch.randn(
            (b, forecaster.n_step_input + forecaster.rollout, 1, g, v),
            dtype=torch.float32,
        ),
    }

    loss, metrics, preds = next(forecaster._rollout_step(batch=batch, rollout=1, validation_mode=False))
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
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

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices={"data": data_indices}, config=_CFG_FORECASTER)
    forecaster.n_step_input = n_step_input
    forecaster.n_step_output = n_step_output
    forecaster.model = DummyModel(num_output_variables=len(data_indices.model.output), output_times=1)
    forecaster.target_dataset_names = forecaster.dataset_names
    forecaster.refresh_dataset_context_static()
    dataset_ctx = forecaster._build_dataset_contexts()["data"]

    b, e, g, v = 1, 1, 2, len(_NAME_TO_INDEX)
    x = torch.zeros((b, forecaster.n_step_input, e, g, v), dtype=torch.float32)
    for step in range(forecaster.n_step_input):
        x[:, step] = float(step + 1)
    y_pred = torch.stack(
        [
            torch.full((b, e, g, v), float(forecaster.n_step_input + step), dtype=torch.float32)
            for step in range(1, forecaster.n_step_output + 1)
        ],
        dim=1,
    )
    batch = torch.zeros(
        (b, forecaster.n_step_input + forecaster.n_step_output, e, g, v),
        dtype=torch.float32,
    )

    updated = forecaster._advance_dataset_input(
        x,
        y_pred,
        batch,
        dataset_ctx=dataset_ctx,
        rollout_step=0,
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


@pytest.mark.parametrize(
    "task_class",
    [GraphMultiOutInterpolator],
    ids=["multi_out"],
)
def test_interpolator_output_times_and_get_init_step(
    task_class: type[GraphMultiOutInterpolator],
) -> None:
    """Both interpolator task types: output_times == len(target), get_init_step(i) == i."""
    interpolator = task_class.__new__(task_class)
    pl.LightningModule.__init__(interpolator)
    interpolator.n_step_input = 1
    interpolator.n_step_output = len(_CFG_INTERP_TWO_TARGETS.training.explicit_times.target)
    interpolator.interp_times = _CFG_INTERP_TWO_TARGETS.training.explicit_times.target
    interpolator.model = None  # unused for this test
    interpolator._plot_adapter = InterpolatorMultiOutPlotAdapter(interpolator)

    assert interpolator.plot_adapter.output_times == 2
    for i in range(interpolator.plot_adapter.output_times):
        assert interpolator.plot_adapter.get_init_step(i) == i


# ---- output_times / get_init_step / _step return format for all tasks ----


def test_graphdiffusionforecaster_output_times_and_get_init_step() -> None:
    """Diffusion has output_times=1 and uses base get_init_step (returns 0)."""
    forecaster = GraphDiffusionForecaster.__new__(GraphDiffusionForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster._plot_adapter = DiffusionPlotAdapter(forecaster)
    assert forecaster.plot_adapter.output_times == 1
    assert forecaster.plot_adapter.get_init_step(0) == 0


def test_graphforecaster_get_init_step() -> None:
    """Forecaster get_init_step(rollout_step) returns 0 for all steps."""
    forecaster = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.rollout = 2
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster._plot_adapter = ForecasterPlotAdapter(forecaster)
    assert forecaster.plot_adapter.get_init_step(0) == 0
    assert forecaster.plot_adapter.get_init_step(1) == 0


def test_graphautoencoder_output_times() -> None:
    """GraphAutoEncoder has output_times=1."""
    from anemoi.training.train.tasks.autoencoder import GraphAutoEncoder

    data_indices = _data_indices_single()
    ae = GraphAutoEncoder.__new__(GraphAutoEncoder)
    pl.LightningModule.__init__(ae)
    _set_base_task_attrs(ae, data_indices=data_indices, config=_CFG_AE)
    ae._plot_adapter = AutoencoderPlotAdapter(ae)
    assert ae.plot_adapter.output_times == 1


def test_graphautoencoder_step_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphAutoEncoder _step returns (loss, metrics, [y_pred]) for consistent task contract."""
    from anemoi.training.train.tasks.autoencoder import GraphAutoEncoder

    data_indices = _data_indices_single()
    ae = GraphAutoEncoder.__new__(GraphAutoEncoder)
    pl.LightningModule.__init__(ae)
    _set_base_task_attrs(ae, data_indices=data_indices, config=_CFG_AE)
    ae.model = DummyModel(
        num_output_variables=len(next(iter(data_indices.values())).model.output),
        output_times=1,
    )
    ae.refresh_dataset_context_static()
    ae.data_indices = {"data": _ExplodingDataIndices()}

    monkeypatch.setattr(
        "torch.utils.checkpoint.checkpoint",
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    monkeypatch.setattr(
        ae,
        "compute_loss_metrics",
        lambda *args, **_kwargs: (torch.tensor(0.0), {}, args[0] if args else None),
    )

    b, t, e, g, v = 2, 1, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, _, y_preds = ae._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


def test_graphmultioutinterpolator_step_returns_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphMultiOutInterpolator _step returns (loss, metrics, [y_pred]) for plot callback contract."""
    data_indices = _data_indices_single()
    task = GraphMultiOutInterpolator.__new__(GraphMultiOutInterpolator)
    pl.LightningModule.__init__(task)
    _set_base_task_attrs(
        task,
        data_indices=data_indices,
        config=_CFG_INTERP_STEP,
        n_step_output=len(_CFG_INTERP_STEP.training.explicit_times.target),
    )
    task.boundary_times = _CFG_INTERP_STEP.training.explicit_times.input
    task.interp_times = _CFG_INTERP_STEP.training.explicit_times.target
    sorted_indices = sorted(set(task.boundary_times + task.interp_times))
    task.imap = {idx: i for i, idx in enumerate(sorted_indices)}
    task.model = DummyModel(
        num_output_variables=len(next(iter(data_indices.values())).model.output),
        output_times=len(task.interp_times),
    )
    task.loss = {"data": DummyLoss()}
    task.refresh_dataset_context_static()
    task.data_indices = {"data": _ExplodingDataIndices()}

    monkeypatch.setattr(
        "torch.utils.checkpoint.checkpoint",
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    monkeypatch.setattr(
        task,
        "compute_loss_metrics",
        lambda *args, **_kwargs: (torch.tensor(0.0), {}, args[0] if args else None),
    )

    b, t, e, g, v = 2, 4, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, _, y_preds = task._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


def test_graphdiffusionforecaster_step_uses_dataset_context_data_indices() -> None:
    class DummyDiffusion:
        def __init__(self, model: DummyDiffusionModel) -> None:
            self.model = model
            self.pre_processors = model.pre_processors
            self.post_processors = model.post_processors

    data_indices = _data_indices_single()
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
    forecaster.refresh_dataset_context_static()
    forecaster.data_indices = {"data": _ExplodingDataIndices()}

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    t = _CFG_DIFFUSION.training.multistep_input
    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)

    loss, _, y_preds = forecaster._step(batch={"data": batch}, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


def test_diffusion_tendency_context_uses_wrapped_tendency_processors() -> None:
    class DummyTendencyCore:
        def __init__(self) -> None:
            self.compute_tendency_calls: list[tuple[Any, Any, Any, Any]] = []
            self.add_tendency_calls: list[tuple[Any, Any, Any, Any]] = []

        def compute_tendency(
            self,
            y: dict[str, Any],
            x_ref: dict[str, Any],
            input_pre_processor: dict[str, Any],
            tendency_pre_processor: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.compute_tendency_calls.append(
                (
                    input_pre_processor["data"],
                    tendency_pre_processor["data"],
                    kwargs["input_post_processor"]["data"],
                    x_ref["data"].shape,
                ),
            )
            return {"data": y["data"]}

        def add_tendency_to_state(
            self,
            _x_ref: dict[str, Any],
            tendency: dict[str, Any],
            output_post_processor: dict[str, Any],
            tendency_post_processor: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.add_tendency_calls.append(
                (
                    output_post_processor["data"],
                    tendency_post_processor["data"],
                    kwargs["output_pre_processor"]["data"],
                    tendency["data"].shape,
                ),
            )
            return {"data": tendency["data"]}

        def _apply_imputer_inverse(
            self,
            post_processors: dict[str, Any],
            dataset_name: str,
            x: dict[str, Any],
        ) -> dict[str, Any]:
            assert dataset_name == "data"
            assert "data" in post_processors
            return x

    class DummyTendencyWrapper:
        def __init__(self) -> None:
            self.model = DummyTendencyCore()
            self.pre_processors = {"data": torch.nn.Identity()}
            self.post_processors = {"data": torch.nn.Identity()}
            self.pre_processors_tendencies = {"data": torch.nn.Identity()}
            self.post_processors_tendencies = {"data": torch.nn.Identity()}

    data_indices = _data_indices_single()
    forecaster = GraphDiffusionTendForecaster.__new__(GraphDiffusionTendForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=_CFG_DIFFUSION)
    forecaster.model = DummyTendencyWrapper()
    forecaster.statistics_tendencies = {"data": {"lead_times": ["6h"], "6h": {}}}
    forecaster._tendency_pre_processors = {}
    forecaster._tendency_post_processors = {}
    forecaster.refresh_dataset_context_static()
    forecaster._validate_tendency_processors()
    forecaster.refresh_dataset_context_static()

    dataset_ctx = forecaster._build_dataset_contexts()["data"]
    assert isinstance(dataset_ctx.static.pre_processor_tendencies, StepwiseProcessors)
    assert isinstance(dataset_ctx.static.post_processor_tendencies, StepwiseProcessors)

    cached_pre = dataset_ctx.static.pre_processor
    cached_post = dataset_ctx.static.post_processor
    cached_pre_tend = dataset_ctx.static.pre_processor_tendencies
    cached_post_tend = dataset_ctx.static.post_processor_tendencies

    forecaster.model.pre_processors = {"data": object()}
    forecaster.model.post_processors = {"data": object()}
    forecaster._tendency_pre_processors = {"data": object()}
    forecaster._tendency_post_processors = {"data": object()}

    y = {"data": torch.ones((2, 1, 1, 3, len(_NAME_TO_INDEX)), dtype=torch.float32)}
    x_ref = {"data": torch.ones((2, 1, 3, 1), dtype=torch.float32)}

    tendency = forecaster._compute_tendency_target(y, x_ref, {"data": dataset_ctx})
    states = forecaster._reconstruct_state(x_ref, tendency, {"data": dataset_ctx})

    assert tendency["data"].shape == y["data"].shape
    assert states["data"].shape == y["data"].shape
    assert forecaster.model.model.compute_tendency_calls[0][0] is cached_pre
    assert forecaster.model.model.compute_tendency_calls[0][1] is cached_pre_tend[0]
    assert forecaster.model.model.compute_tendency_calls[0][2] is cached_post
    assert forecaster.model.model.add_tendency_calls[0][0] is cached_post
    assert forecaster.model.model.add_tendency_calls[0][1] is cached_post_tend[0]
    assert forecaster.model.model.add_tendency_calls[0][2] is cached_pre
