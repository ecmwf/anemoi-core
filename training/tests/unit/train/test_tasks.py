from typing import Any

import einops
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
from anemoi.training.losses.index_space import IndexSpace
from anemoi.training.losses.loss import get_loss_function
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.train.tasks.diffusionforecaster import GraphDiffusionForecaster
from anemoi.training.train.tasks.ensforecaster import GraphEnsForecaster
from anemoi.training.train.tasks.forecaster import GraphForecaster
from anemoi.training.train.tasks.interpolator import GraphInterpolator
from anemoi.training.train.tasks.interpolator import GraphMultiOutInterpolator
from anemoi.training.utils.masks import NoOutputMask


class DummyLoss(torch.nn.Module):

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.mean((y_pred - y) ** 2)


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


def test_graphinterpolator_preserves_time_dim_in_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: interpolator loss targets must keep singleton time dim."""
    forecaster = GraphInterpolator.__new__(GraphInterpolator)
    pl.LightningModule.__init__(forecaster)

    name_to_index = {"A": 0, "B": 1}
    data_indices = _make_minimal_index_collection(name_to_index)
    forecaster.data_indices = {"data": data_indices}
    forecaster.dataset_names = ["data"]
    forecaster.boundary_times = [0, 2]
    forecaster.interp_times = [1]
    forecaster.imap = {0: 0, 1: 1, 2: 2}
    forecaster.n_step_output = 1
    forecaster.rollout = 1
    forecaster.num_tfi = {"data": 0}
    forecaster.use_time_fraction = {"data": False}
    forecaster.target_forcing_indices = {"data": []}

    def _forward_stub(
        self: GraphInterpolator,
        x_bound: dict[str, torch.Tensor],
        target_forcing: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del self, target_forcing
        x = x_bound["data"]
        b, _, e, g, _ = x.shape
        return {"data": torch.randn((b, 1, e, g, len(name_to_index)), dtype=x.dtype, device=x.device)}

    def _compute_loss_metrics_stub(
        self: GraphInterpolator,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del self, kwargs
        assert y["data"].ndim == 5
        assert y["data"].shape[1] == 1
        assert y_pred["data"].shape == y["data"].shape
        return torch.tensor(0.0), {}, y_pred

    monkeypatch.setattr(GraphInterpolator, "forward", _forward_stub, raising=True)
    monkeypatch.setattr(GraphInterpolator, "compute_loss_metrics", _compute_loss_metrics_stub, raising=True)

    b, e, g, v = 3, 1, 4, len(name_to_index)  # b>1 to guard against silent broadcast bugs
    batch = {"data": torch.randn((b, 3, e, g, v), dtype=torch.float32)}

    loss, metrics, y_preds = forecaster._step(batch=batch, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert len(y_preds) == 1
    assert y_preds[0]["data"].shape == (b, 1, e, g, v)


def test_graphmultioutinterpolator_uses_data_output_target_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-out interpolator should label pre-sliced targets as DATA_OUTPUT."""
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
        assert kwargs["target_layout"] == IndexSpace.DATA_OUTPUT
        assert y["data"].shape[-1] == len(data_indices.data.output.full)
        return torch.tensor(0.0), {}, y

    monkeypatch.setattr(GraphMultiOutInterpolator, "forward", _forward_stub, raising=True)
    monkeypatch.setattr(GraphMultiOutInterpolator, "compute_loss_metrics", _compute_loss_metrics_stub, raising=True)

    b, e, g, v = 2, 1, 4, len(data_indices.name_to_index)
    batch = {"data": torch.randn((b, 3, e, g, v), dtype=torch.float32)}

    loss, metrics, y_pred = forecaster._step(batch=batch, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert y_pred["data"].shape == (b, 2, e, g, 2)


def test_calculate_val_metrics_rejects_variable_scaled_filtered_metric() -> None:
    """Validation metrics must reject variable-dimension scaling even through wrappers."""
    module = GraphInterpolator.__new__(GraphInterpolator)
    pl.LightningModule.__init__(module)

    name_to_index = {"A": 0, "B": 1}
    data_indices = _make_minimal_index_collection(name_to_index)

    metric = get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.MSELoss", "scalers": ["grid_uniform", "double_weight"]}),
        scalers={
            "grid_uniform": (3, torch.ones(4)),
            "double_weight": (4, torch.ones(len(data_indices.model.output.full)) * 2.0),
        },
        data_indices=data_indices,
    )

    class _Model:
        def __init__(self) -> None:
            def _identity_post_processor(x: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
                return x

            self.post_processors = {"data": _identity_post_processor}

    module.model = _Model()
    module.metrics = {"data": {"mse": metric}}
    module.val_metric_ranges = {"data": {"all": [0, 1]}}
    module.model_comm_group = None
    module.model_comm_group_size = 1
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": None}

    y_pred = torch.randn(2, 1, 1, 4, 2)
    y = torch.randn(2, 1, 1, 4, 2)
    with pytest.raises(ValueError, match="Validation metrics cannot be scaled over the variable dimension"):
        module.calculate_val_metrics(
            y_pred=y_pred,
            y=y,
            dataset_name="data",
            pred_layout=IndexSpace.MODEL_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )


def test_calculate_val_metrics_rejects_non_baseloss_metric() -> None:
    """Validation metrics must be BaseLoss instances to ensure consistent filtering/remapping."""
    module = GraphInterpolator.__new__(GraphInterpolator)
    pl.LightningModule.__init__(module)

    class _Model:
        def __init__(self) -> None:
            def _identity_post_processor(x: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
                return x

            self.post_processors = {"data": _identity_post_processor}

    module.model = _Model()
    module.metrics = {"data": {"custom": lambda *_args, **_kwargs: torch.tensor(0.0)}}
    module.val_metric_ranges = {"data": {"all": [0, 1]}}
    module.model_comm_group = None
    module.model_comm_group_size = 1
    module.grid_dim = -2
    module.grid_shard_shapes = {"data": None}

    y_pred = torch.randn(2, 1, 1, 4, 2)
    y = torch.randn(2, 1, 1, 4, 2)
    with pytest.raises(AssertionError, match="must inherit BaseLoss"):
        module.calculate_val_metrics(
            y_pred=y_pred,
            y=y,
            dataset_name="data",
            pred_layout=IndexSpace.MODEL_OUTPUT,
            target_layout=IndexSpace.DATA_FULL,
        )


def test_update_scalers_applies_to_filtered_loss_wrapper() -> None:
    """Updating scalers must work for LossVariableMapper-backed losses."""
    module = GraphInterpolator.__new__(GraphInterpolator)
    pl.LightningModule.__init__(module)

    data_indices = _make_minimal_index_collection({"A": 0, "B": 1})
    loss = get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.MSELoss", "scalers": ["dynamic"]}),
        scalers={"dynamic": (4, torch.ones(2))},
        data_indices=data_indices,
    )

    class _Updater:
        def update_scaling_values(self, callback: AvailableCallbacks, **kwargs: Any) -> tuple[tuple[int], torch.Tensor]:
            del callback, kwargs
            return (4,), torch.tensor([3.0, 5.0])

    module.model = object()
    module._update_scaler_for_dataset(
        name="dynamic",
        scaler_builder=_Updater(),
        callback=AvailableCallbacks.ON_BATCH_START,
        loss_obj=loss,
        metrics_dict={},
        dataset_name="data",
    )

    updated = loss.loss.scaler.tensors["dynamic"][1]
    torch.testing.assert_close(updated, torch.tensor([3.0, 5.0]))


def test_graphensforecaster_compute_dataset_loss_metrics_forwards_layout_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphEnsForecaster must forward pred/target layout metadata to loss and metrics."""
    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)

    forecaster.ens_comm_subgroup_size = 1
    forecaster.ens_comm_subgroup = None
    forecaster.grid_shard_slice = {"data": None}
    forecaster.grid_dim = -2
    forecaster.grid_shard_shapes = {"data": None}

    # Avoid distributed collective in this unit test.
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

    def _compute_loss_metrics_stub(
        self: GraphEnsForecaster,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        step: int | None = None,
        validation_mode: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        del y, step, validation_mode, kwargs
        assert isinstance(self, GraphEnsForecaster)
        pred = next(iter(y_pred.values()))
        return torch.zeros(1, device=pred.device, dtype=pred.dtype), {}, y_pred

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


@pytest.mark.parametrize("task_class", [GraphInterpolator, GraphMultiOutInterpolator], ids=["single_out", "multi_out"])
def test_interpolator_output_times_and_get_init_step(
    task_class: type[GraphInterpolator] | type[GraphMultiOutInterpolator],
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


def test_graphinterpolator_step_returns_list_of_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphInterpolator (single-out) _step returns (loss, metrics, list of dicts) per interp step."""

    def dummy_forward(x_bound: dict, target_forcing: dict | None = None) -> dict:
        del target_forcing
        b = next(iter(x_bound.values())).shape[0]
        e = next(iter(x_bound.values())).shape[2]
        g = next(iter(x_bound.values())).shape[3]
        v = next(iter(x_bound.values())).shape[4]
        return {"data": torch.randn(b, 1, e, g, v, dtype=torch.float32)}

    data_indices = _data_indices_single()
    task = GraphInterpolator.__new__(GraphInterpolator)
    pl.LightningModule.__init__(task)
    _set_base_task_attrs(task, data_indices=data_indices, config=_CFG_INTERP_STEP)
    task.interp_times = _CFG_INTERP_STEP.training.explicit_times.target
    task.boundary_times = _CFG_INTERP_STEP.training.explicit_times.input
    sorted_indices = sorted(set(task.boundary_times + task.interp_times))
    task.imap = {idx: i for i, idx in enumerate(sorted_indices)}
    task.num_tfi = {"data": 0}
    task.use_time_fraction = {"data": False}
    task.target_forcing_indices = {"data": []}
    task.model = type(
        "M",
        (),
        {"__call__": lambda _self, x_bound, target_forcing=None, **_kwargs: dummy_forward(x_bound, target_forcing)},
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

    _assert_step_return_format(loss, y_preds, expected_len=len(task.interp_times))


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
