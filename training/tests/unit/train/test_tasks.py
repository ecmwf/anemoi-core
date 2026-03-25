from typing import Any

import einops
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.train.tasks.diffusionforecaster import GraphDiffusionForecaster
from anemoi.training.train.tasks.ensforecaster import GraphEnsForecaster
from anemoi.training.train.tasks.forecaster import GraphForecaster
from anemoi.training.train.tasks.interpolator import GraphMultiOutInterpolator
from anemoi.training.train.tasks.nowcaster import GraphNowcaster
from anemoi.training.utils.masks import NoOutputMask


class DummyLoss(torch.nn.Module):

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.mean((y_pred - y) ** 2)


class RecordingLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.seen_pred: torch.Tensor | None = None
        self.seen_target: torch.Tensor | None = None

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self.seen_pred = y_pred.detach().clone()
        self.seen_target = y.detach().clone()
        return torch.mean((y_pred - y) ** 2)


class DummyCompositeLoss(torch.nn.Module):
    """CombinedLoss-like object: supports update_scaler but has no .scaler attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.updated: list[str] = []

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        del scaler, override
        self.updated.append(name)


class DummyLossWithComponents(torch.nn.Module):

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.mean((y_pred - y) ** 2)

    def component_metrics(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        loss = torch.mean((y_pred - y) ** 2)
        return {
            "loss_component_demo_raw": loss,
            "loss_component_demo_scaled": loss * 2.0,
            "loss_component_demo_weighted": loss * 3.0,
        }


class DummyScalerBuilder:

    def update_scaling_values(self, callback: AvailableCallbacks, **kwargs):
        del callback, kwargs
        return ((), torch.tensor([1.0]))


class DummyMetricWithScaler(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.scaler = {"nan_mask_weights": torch.tensor([1.0])}
        self.updated: list[str] = []

    def update_scaler(self, name: str, scaler: torch.Tensor, *, override: bool = False) -> None:
        del scaler, override
        self.updated.append(name)


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


class ShiftProcessor(torch.nn.Module):

    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        del in_place, kwargs
        return x + self.shift


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

    assert forecaster.output_times == 1
    for i in range(1, _CFG_FORECASTER.training.rollout.max + 1):
        forecaster.rollout = i
        assert forecaster.get_init_step(i) == 0
        assert forecaster.output_times == i

    # _step returns one prediction per rollout step with shape (B, n_step_output, E, G, V)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
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


def test_compute_dataset_loss_metrics_postprocesses_training_loss() -> None:
    data_indices = _data_indices_single()
    config = DictConfig({"training": {"training_loss_postprocessed": True}})
    forecaster = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=config)
    forecaster.training_loss_postprocessed = True
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True
    forecaster.model = type("M", (), {"post_processors": {"data": ShiftProcessor(10.0)}})()

    loss_fn = RecordingLoss()
    forecaster.loss = {"data": loss_fn}

    y_pred = torch.zeros((1, 1, 1, 1, len(_NAME_TO_INDEX)), dtype=torch.float32)
    y = torch.ones_like(y_pred)

    loss, metrics, original_pred = forecaster.compute_dataset_loss_metrics(
        y_pred,
        y,
        validation_mode=False,
        dataset_name="data",
    )

    assert torch.isclose(loss, torch.tensor(1.0))
    assert metrics == {}
    assert torch.equal(original_pred, y_pred)
    assert loss_fn.seen_pred is not None
    assert loss_fn.seen_target is not None
    assert torch.allclose(loss_fn.seen_pred, y_pred + 10.0)
    assert torch.allclose(loss_fn.seen_target, y + 10.0)


def test_compute_dataset_loss_metrics_includes_loss_component_metrics_in_validation() -> None:
    data_indices = _data_indices_single()
    config = DictConfig({"training": {"training_loss_postprocessed": False}})
    forecaster = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(forecaster)
    _set_base_task_attrs(forecaster, data_indices=data_indices, config=config)
    forecaster.training_loss_postprocessed = False
    forecaster.loss_supports_sharding = False
    forecaster.metrics_support_sharding = True
    forecaster.model = type("M", (), {"post_processors": {"data": ShiftProcessor(0.0)}})()
    forecaster.loss = {"data": DummyLossWithComponents()}
    forecaster.metrics = {"data": torch.nn.ModuleDict()}
    forecaster.val_metric_ranges = {"data": {"all": [0, 1]}}

    y_pred = torch.zeros((1, 1, 1, 1, len(_NAME_TO_INDEX)), dtype=torch.float32)
    y = torch.ones_like(y_pred)

    loss, metrics, _ = forecaster.compute_dataset_loss_metrics(
        y_pred,
        y,
        validation_mode=True,
        dataset_name="data",
    )

    assert torch.isclose(loss, torch.tensor(1.0))
    assert torch.allclose(metrics["loss_component_demo_raw"], torch.tensor(1.0))
    assert torch.allclose(metrics["loss_component_demo_scaled"], torch.tensor(2.0))
    assert torch.allclose(metrics["loss_component_demo_weighted"], torch.tensor(3.0))


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

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    t = _CFG_DIFFUSION.training.multistep_input

    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)
    loss, _, y_preds = forecaster._step(batch={"data": batch}, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)
    y_pred = y_preds[0]["data"]
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, 1, e, g, v)


def test_graphensforecaster_rollout_with_time_dim_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollout step works when model returns (B, T, E, G, V); _advance_input uses last time step."""
    data_indices = _make_minimal_index_collection(_NAME_TO_INDEX)

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    forecaster.rollout = 1
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
    batch = {"data": torch.randn((b, forecaster.n_step_input + forecaster.rollout, 1, g, v), dtype=torch.float32)}

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
    forecaster.n_step_input = n_step_input
    forecaster.n_step_output = n_step_output
    forecaster.output_mask = {"data": NoOutputMask()}
    forecaster.data_indices = {"data": data_indices}
    forecaster.grid_shard_slice = {"data": None}

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
    batch = torch.zeros((b, forecaster.n_step_input + forecaster.n_step_output, e, g, v), dtype=torch.float32)

    updated = forecaster._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        dataset_name="data",
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


def test_update_scaler_for_dataset_accepts_composite_loss_without_scaler() -> None:
    """Composite losses (e.g. CombinedLoss) can update scalers even without a `.scaler` attr."""
    task = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(task)
    task.model = DummyModel()

    loss_obj = DummyCompositeLoss()
    metric = DummyMetricWithScaler()

    task._update_scaler_for_dataset(
        name="nan_mask_weights",
        scaler_builder=DummyScalerBuilder(),
        callback=AvailableCallbacks.ON_BATCH_START,
        loss_obj=loss_obj,
        metrics_dict={"metric": metric},
        dataset_name="data",
    )

    assert loss_obj.updated == ["nan_mask_weights"]
    assert metric.updated == ["nan_mask_weights"]


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


@pytest.mark.parametrize("task_class", [GraphMultiOutInterpolator], ids=["multi_out"])
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
    interpolator.get_init_step = lambda rollout: rollout

    assert interpolator.output_times == 2
    for i in range(interpolator.output_times):
        assert interpolator.get_init_step(i) == i


# ---- output_times / get_init_step / _step return format for all tasks ----


def test_graphdiffusionforecaster_output_times_and_get_init_step() -> None:
    """Diffusion has output_times=1 and uses base get_init_step (returns 0)."""
    from anemoi.training.train.tasks.diffusionforecaster import GraphDiffusionForecaster

    forecaster = GraphDiffusionForecaster.__new__(GraphDiffusionForecaster)
    pl.LightningModule.__init__(forecaster)
    assert forecaster.output_times == 1
    assert forecaster.get_init_step(0) == 0


def test_graphforecaster_get_init_step() -> None:
    """Forecaster get_init_step(rollout_step) returns 0 for all steps."""
    forecaster = GraphForecaster.__new__(GraphForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.rollout = 2
    forecaster.n_step_input = 1
    forecaster.n_step_output = 1
    assert forecaster.get_init_step(0) == 0
    assert forecaster.get_init_step(1) == 0


def test_graphautoencoder_output_times() -> None:
    """GraphAutoEncoder has output_times=1."""
    from anemoi.training.train.tasks.autoencoder import GraphAutoEncoder

    data_indices = _data_indices_single()
    ae = GraphAutoEncoder.__new__(GraphAutoEncoder)
    pl.LightningModule.__init__(ae)
    _set_base_task_attrs(ae, data_indices=data_indices, config=_CFG_AE)
    assert ae.output_times == 1


def test_graphautoencoder_step_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphAutoEncoder _step returns (loss, metrics, [y_pred]) for consistent task contract."""
    from anemoi.training.train.tasks.autoencoder import GraphAutoEncoder

    def dummy_forward(x: dict) -> dict:
        b = next(iter(x.values())).shape[0]
        t = next(iter(x.values())).shape[1]
        e = next(iter(x.values())).shape[2]
        g = next(iter(x.values())).shape[3]
        v = next(iter(x.values())).shape[4]
        return {dn: torch.randn(b, t, e, g, v, dtype=torch.float32) for dn in x}

    data_indices = _data_indices_single()
    ae = GraphAutoEncoder.__new__(GraphAutoEncoder)
    pl.LightningModule.__init__(ae)
    _set_base_task_attrs(ae, data_indices=data_indices, config=_CFG_AE)
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


def test_graphmultioutinterpolator_step_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphMultiOutInterpolator _step returns (loss, metrics, [y_pred]) for plot callback contract."""

    def dummy_forward(x_bound: dict) -> dict:
        b = next(iter(x_bound.values())).shape[0]
        e = next(iter(x_bound.values())).shape[2]
        g = next(iter(x_bound.values())).shape[3]
        v = next(iter(x_bound.values())).shape[4]
        return {"data": torch.randn(b, 2, e, g, v, dtype=torch.float32)}

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
    task.model = type(
        "M",
        (),
        {"__call__": lambda _self, x, **_kwargs: dummy_forward(x)},
    )()
    task.loss = {"data": DummyLoss()}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(
        task,
        "compute_loss_metrics",
        lambda *args, **_kwargs: (torch.tensor(0.0), {}, args[0] if args else None),
    )

    b, t, e, g, v = 2, 4, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, _, y_preds = task._step(batch, validation_mode=False)

    _assert_step_return_format(loss, y_preds, expected_len=1)


def test_graphnowcaster_build_decoder_context_time_only() -> None:
    """Nowcaster time-only decoder conditioning builds a single time-fraction channel."""

    task = GraphNowcaster.__new__(GraphNowcaster)
    pl.LightningModule.__init__(task)
    task.dataset_names = ["opera"]
    task.decoder_context_sources = {"opera": ["meps"]}
    task.decoder_context_time_only = True
    task.n_step_output = 3
    task.interp_times = [6, 8, 11]
    task.imap = {idx: idx for idx in range(13)}
    task.dataset_time_maps = {
        "opera": {idx: idx for idx in range(13)},
        "meps": {idx: idx for idx in range(13)},
    }

    batch = {
        "opera": torch.zeros(2, 13, 1, 4, 3, dtype=torch.float32),
        "meps": torch.ones(2, 13, 1, 2, 3, dtype=torch.float32),
    }

    ctx = task._build_decoder_context(batch, decode_dataset_names=("opera",))

    cond = ctx["opera"]["cond"]
    assert cond.shape == (2, 3, 1, 4, 1)
    expected = torch.tensor([(6 - 6) / (11 - 6), (8 - 6) / (11 - 6), (11 - 6) / (11 - 6)], dtype=cond.dtype)
    assert torch.allclose(cond[:, :, 0, 0, 0], expected.unsqueeze(0).expand(2, -1))


def test_graphnowcaster_required_target_time_must_exist() -> None:
    """Nowcaster requires exact target-time availability for decoded datasets."""

    task = GraphNowcaster.__new__(GraphNowcaster)
    pl.LightningModule.__init__(task)
    task.imap = {0: 0, 1: 1}
    task.dataset_time_maps = {"opera": {0: 0, 2: 1}}

    with pytest.raises(ValueError, match="missing exact target time 1"):
        task._lookup_required_time_indices(
            dataset_name="opera",
            relative_times=[1],
            usage="target",
        )


def test_graphnowcaster_step_uses_configured_input_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nowcaster selects encoder inputs from configured model-relative input times."""

    task = GraphNowcaster.__new__(GraphNowcaster)
    pl.LightningModule.__init__(task)
    data_indices = _data_indices_single()
    _set_base_task_attrs(task, data_indices=data_indices, config=_CFG_INTERP_STEP, n_step_input=2, n_step_output=1)
    task.target_dataset_names = ["data"]
    task.dataset_names = ["data"]
    task.input_times = [2, 4]
    task.interp_times = [5]
    task.dataset_time_maps = {"data": {2: 0, 4: 1, 5: 2}}
    task.decoder_context_time_only = True
    task._timing_rank0_only = True
    task._timing_step_enabled = False
    task._timing_step_every = 1
    task._timing_model_enabled = False
    task._timing_model_every = 1
    task._timing_step_counters = {"train": 0, "val": 0}
    task._timing_model_counters = {"train": 0, "val": 0}

    captured: dict[str, torch.Tensor] = {}

    class _Model:
        def __call__(self, x: dict[str, torch.Tensor], **_kwargs):
            captured["x"] = x["data"].detach().clone()
            b, _t, e, g, v = x["data"].shape
            return {"data": torch.zeros((b, 1, e, g, v), dtype=x["data"].dtype)}

    task.model = _Model()
    task.loss = {"data": DummyLoss()}

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(
        task,
        "compute_loss_metrics",
        lambda *args, **_kwargs: (torch.tensor(0.0), {}, args[0] if args else None),
    )

    batch = {"data": torch.zeros((1, 3, 1, 1, 2), dtype=torch.float32)}
    batch["data"][:, 0, ...] = 10.0
    batch["data"][:, 1, ...] = 20.0
    batch["data"][:, 2, ...] = 30.0

    _loss, _metrics, _y_pred = task._step(batch, validation_mode=False)

    assert "x" in captured
    assert captured["x"].shape == (1, 2, 1, 1, 2)
    assert torch.allclose(captured["x"][:, 0, ...], torch.full((1, 1, 1, 2), 10.0))
    assert torch.allclose(captured["x"][:, 1, ...], torch.full((1, 1, 1, 2), 20.0))


def test_graphnowcaster_available_target_time_indices() -> None:
    """Nowcaster selects only available target times for sparse per-dataset supervision."""

    task = GraphNowcaster.__new__(GraphNowcaster)
    pl.LightningModule.__init__(task)
    task.imap = {0: 0, 1: 1, 2: 2}
    task.interp_times = [1, 2, 3]
    task.dataset_time_maps = {"opera": {1: 0, 3: 1}}

    pred_indices, batch_indices = task._lookup_available_target_time_indices(dataset_name="opera")

    assert pred_indices == [0, 2]
    assert batch_indices == [0, 1]


def test_graphnowcaster_build_decoder_context_time_fraction_only() -> None:
    """Decoder context is always time-fraction-only, regardless of source sparsity."""

    task = GraphNowcaster.__new__(GraphNowcaster)
    pl.LightningModule.__init__(task)
    task.dataset_names = ["opera", "meps"]
    task.decoder_context_sources = {"opera": ["meps"]}
    task.decoder_context_time_only = True
    task.n_step_output = 2
    task.interp_times = [1, 2]
    task.imap = {0: 0, 1: 1, 2: 2, 4: 3}
    task.dataset_time_maps = {
        "opera": {0: 0, 1: 1, 2: 2, 4: 3},
        "meps": {0: 0, 2: 1},  # No exact value at relative time 1.
    }

    opera = torch.zeros(1, 4, 1, 2, 1, dtype=torch.float32)
    meps = torch.zeros(1, 2, 1, 2, 1, dtype=torch.float32)
    meps[:, 0, ...] = 10.0
    meps[:, 1, ...] = 20.0
    batch = {"opera": opera, "meps": meps}

    ctx = task._build_decoder_context(batch, decode_dataset_names=("opera",))
    cond = ctx["opera"]["cond"]

    # Time-fraction channel is always the only conditioning channel.
    assert cond.shape == (1, 2, 1, 2, 1)
    expected_time_fraction = torch.tensor([0.0, 1.0], dtype=cond.dtype)
    assert torch.allclose(cond[0, :, 0, 0, 0], expected_time_fraction)
