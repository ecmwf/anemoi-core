from collections.abc import Callable
from typing import Any

import einops
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import Processors
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


def _stub_init_forecaster(
    self: BaseGraphModule,
    *,
    config: DictConfig,
    graph_data: HeteroData,
    statistics: dict,
    statistics_tendencies: dict,
    data_indices: IndexCollection,
    metadata: dict | None = None,
    supporting_arrays: dict | None = None,
) -> None:
    del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
    pl.LightningModule.__init__(self)
    self.n_step_input = config.training.multistep_input
    self.n_step_output = config.training.multistep_output
    self.model = DummyModel(num_output_variables=len(next(iter(data_indices.values())).model.output))
    self.model_comm_group = None
    self.model_comm_group_size = 1
    self.grid_shard_shapes = {"data": None}
    self.grid_shard_slice = {"data": None}
    self.is_first_step = False
    self.updating_scalars = {}
    self.data_indices = data_indices
    self.dataset_names = list(data_indices.keys())
    self.target_dataset_names = self.dataset_names
    self.grid_dim = -2
    self.config = config
    self.loss = {"data": DummyLoss()}
    self.loss_supports_sharding = False
    self.metrics_support_sharding = True
    self.device = torch.device("cpu")


def _make_graph_forecaster(monkeypatch: pytest.MonkeyPatch) -> GraphForecaster:
    """Build GraphForecaster with stubbed BaseGraphModule.__init__ (single place for forecaster creation)."""
    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init_forecaster, raising=True)
    return GraphForecaster(
        config=_CFG_FORECASTER,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=_data_indices_single(),
        metadata={},
        supporting_arrays={},
    )


def test_graphforecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecaster output_times, get_init_step, and _step return shape (one instantiation)."""
    forecaster = _make_graph_forecaster(monkeypatch)

    assert forecaster.output_times == 1
    for i in range(1, _CFG_FORECASTER.training.rollout.max + 1):
        forecaster.rollout = i
        assert forecaster.get_init_step(i) == 0
        assert forecaster.output_times == i

    # _step returns one prediction per rollout step with shape (B, n_step_output, E, G, V)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(forecaster, "_advance_input", lambda x: x)

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


def test_graphdiffusionforecaster(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDiffusion:
        def __init__(self, model: DummyDiffusionModel) -> None:
            self.model = model

    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.n_step_input = config.training.multistep_input
        model = DummyDiffusionModel(num_output_variables=len(next(iter(data_indices.values())).model.output))
        self.model = DummyDiffusion(model=model)
        self.model_comm_group = None
        self.model_comm_group_size = 1
        self.grid_shard_shapes = {"data": None}
        self.grid_shard_slice = {"data": None}
        self.is_first_step = False
        self.updating_scalars = {}
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.target_dataset_names = self.dataset_names
        self.grid_dim = -2
        self.config = config
        self.loss = {"data": DummyLoss()}
        self.loss_supports_sharding = False
        self.metrics_support_sharding = True
        self.n_step_output = 1

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)

    forecaster = GraphDiffusionForecaster(
        config=_CFG_DIFFUSION,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=_data_indices_single(),
        metadata={},
        supporting_arrays={},
    )

    b, e, g, v = 2, 1, 4, len(_NAME_TO_INDEX)
    t = _CFG_DIFFUSION.training.multistep_input

    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)
    loss, metrics, y_preds = forecaster._step(batch={"data": batch}, validation_mode=False)

    _assert_step_return_format(loss, metrics, y_preds, expected_len=1)
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


def _make_stub_init_autoencoder(model: Any = None) -> Callable[[BaseGraphModule], None]:
    """Return a stub __init__ for GraphAutoEncoder. If model is set, stub sets self.model (for _step tests)."""

    def _stub(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.n_step_input = 1
        self.n_step_output = 1
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.config = config
        if model is not None:
            self.model = model

    return _stub


@pytest.mark.parametrize("task_class", [GraphInterpolator, GraphMultiOutInterpolator], ids=["single_out", "multi_out"])
def test_interpolator_output_times_and_get_init_step(
    monkeypatch: pytest.MonkeyPatch,
    task_class: type[GraphInterpolator] | type[GraphMultiOutInterpolator],
) -> None:
    """Both interpolator task types: output_times == len(target), get_init_step(i) == i under stub."""
    import types

    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, data_indices, supporting_arrays
        self.n_step_input = 1
        self.n_step_output = len(config.training.explicit_times.target)
        self.interp_times = config.training.explicit_times.target
        self.model = types.SimpleNamespace()
        self.get_init_step = lambda rollout: rollout

    monkeypatch.setattr(task_class, "__init__", _stub_init, raising=True)

    data_indices = {"data": _DummyIndexForInterpolator()}
    interpolator = task_class(
        config=_CFG_INTERP_TWO_TARGETS,
        graph_data={"data": None},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=data_indices,
        metadata={},
        supporting_arrays={},
    )

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


def test_graphautoencoder_output_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphAutoEncoder has output_times=1."""
    from anemoi.training.train.tasks.autoencoder import GraphAutoEncoder

    monkeypatch.setattr(BaseGraphModule, "__init__", _make_stub_init_autoencoder(), raising=True)
    ae = GraphAutoEncoder(
        config=_CFG_AE,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=_data_indices_single(),
        metadata={},
        supporting_arrays={},
    )
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

    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.n_step_input = 1
        self.n_step_output = 1
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.config = config
        self.model = type("M", (), {"__call__": lambda _self, x: dummy_forward(x)})()

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    ae = GraphAutoEncoder(
        config=_CFG_AE,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=_data_indices_single(),
        metadata={},
        supporting_arrays={},
    )
    monkeypatch.setattr(ae, "compute_loss_metrics", lambda y_pred: (torch.tensor(0.0), {}, y_pred))

    b, t, e, g, v = 2, 1, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, metrics, y_preds = ae._step(batch, validation_mode=False)

    _assert_step_return_format(loss, metrics, y_preds, expected_len=1)


def test_graphinterpolator_step_returns_list_of_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphInterpolator (single-out) _step returns (loss, metrics, list of dicts) per interp step."""

    def dummy_forward(x_bound: dict) -> dict:
        b = next(iter(x_bound.values())).shape[0]
        e = next(iter(x_bound.values())).shape[2]
        g = next(iter(x_bound.values())).shape[3]
        v = next(iter(x_bound.values())).shape[4]
        return {"data": torch.randn(b, 1, e, g, v, dtype=torch.float32)}

    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.n_step_input = 1
        self.n_step_output = 1
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.config = config
        self.interp_times = config.training.explicit_times.target
        self.boundary_times = config.training.explicit_times.input
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}
        self.num_tfi = {"data": 0}
        self.use_time_fraction = {"data": False}
        self.target_forcing_indices = {"data": []}
        self.model = type(
            "M",
            (),
            {"__call__": lambda _self, x_bound, target_forcing=None: dummy_forward(x_bound, target_forcing)},
        )()
        self.loss = {"data": DummyLoss()}
        self.grid_dim = -2
        self.device = torch.device("cpu")

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    task = GraphInterpolator(
        config=_CFG_INTERP_STEP,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=_data_indices_single(),
        metadata={},
        supporting_arrays={},
    )

    def _fake_compute(y_pred: torch.Tensor) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        return torch.tensor(0.0), {}, y_pred

    monkeypatch.setattr(task, "compute_loss_metrics", _fake_compute)

    b, t, e, g, v = 2, 4, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, metrics, y_preds = task._step(batch, validation_mode=False)

    _assert_step_return_format(loss, metrics, y_preds, expected_len=len(task.interp_times))


def test_graphmultioutinterpolator_step_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """GraphMultiOutInterpolator _step returns (loss, metrics, [y_pred]) for plot callback contract."""

    def dummy_forward(x_bound: dict) -> dict:
        b = next(iter(x_bound.values())).shape[0]
        e = next(iter(x_bound.values())).shape[2]
        g = next(iter(x_bound.values())).shape[3]
        v = next(iter(x_bound.values())).shape[4]
        return {"data": torch.randn(b, 2, e, g, v, dtype=torch.float32)}

    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.n_step_input = 1
        self.n_step_output = len(config.training.explicit_times.target)
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.config = config
        self.boundary_times = config.training.explicit_times.input
        self.interp_times = config.training.explicit_times.target
        sorted_indices = sorted(set(self.boundary_times + self.interp_times))
        self.imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}
        self.model = type("M", (), {"__call__": lambda _self, x: dummy_forward(x)})()
        self.loss = {"data": DummyLoss()}
        self.grid_dim = -2
        self.device = torch.device("cpu")

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    task = GraphMultiOutInterpolator(
        config=_CFG_INTERP_STEP,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=_data_indices_single(),
        metadata={},
        supporting_arrays={},
    )

    def _fake_compute(y_pred: torch.Tensor) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        return torch.tensor(0.0), {}, y_pred

    monkeypatch.setattr(task, "compute_loss_metrics", _fake_compute)

    b, t, e, g, v = 2, 4, 1, 4, 2
    batch = {"data": torch.randn(b, t, e, g, v, dtype=torch.float32)}
    loss, metrics, y_preds = task._step(batch, validation_mode=False)

    _assert_step_return_format(loss, metrics, y_preds, expected_len=1)
