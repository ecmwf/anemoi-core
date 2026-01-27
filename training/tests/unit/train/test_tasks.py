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
from anemoi.training.train.tasks.interpolator import GraphInterpolator
from anemoi.training.train.tasks.obsinterpolator import ObsGraphInterpolator
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


def test_graphinterpolator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.model = DummyModel(output_times=len(config.training.explicit_times.target), add_skip=True)
        self.model_comm_group = None
        self.grid_shard_slice = {"data": None}
        self.grid_shard_shapes = {"data": None}
        self.is_first_step = False  # avoid scaler init
        self.updating_scalars = {}
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.config = config
        self.loss = {"data": DummyLoss()}
        self.loss_supports_sharding = True
        self.metrics_support_sharding = True
        self.grid_dim = -2
        self.multi_out = 1

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    name_to_index = {"A": 0, "B": 1}
    itp = GraphInterpolator.__new__(GraphInterpolator)
    itp = GraphInterpolator(
        config=DictConfig(
            {
                "training": {
                    "explicit_times": {"input": [0, 6], "target": [1, 2, 3]},
                    "target_forcing": {"data": [], "time_fraction": False},
                },
            },
        ),
        graph_data={"data": HeteroData()},
        statistics={},
        statistics_tendencies={},
        data_indices={"data": _make_minimal_index_collection(name_to_index)},
        metadata={},
        supporting_arrays={},
    )
    assert itp.multi_out == 1
    assert itp.boundary_times == [0, 6]
    assert itp.interp_times == [1, 2, 3]
    b, e, g, v = 2, 1, 3, len(name_to_index)
    t = len(itp.imap)
    batch = torch.randn((b, e, g, v), dtype=torch.float32)
    loss, metrics, y_preds = itp._step(batch={"data": batch}, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert len(y_preds) == len(itp.interp_times)
    y_pred = torch.stack([pred["data"] for pred in y_preds], dim=1)
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, len(itp.interp_times), e, g, v)


def test_graphmultioutinterpolator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.model = DummyModel(output_times=len(config.training.explicit_times.target), add_skip=True)
        self.model_comm_group = None
        self.grid_shard_slice = {"data": None}
        self.grid_shard_shapes = {"data": None}
        self.is_first_step = False  # avoid scaler init
        self.updating_scalars = {}
        self.data_indices = data_indices
        self.dataset_names = list(data_indices.keys())
        self.config = config
        self.loss = {"data": DummyLoss()}
        self.loss_supports_sharding = True
        self.metrics_support_sharding = True
        self.grid_dim = -2
        self.multi_out = config.training.multistep_output

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    name_to_index = {"A": 0, "B": 1}
    itp = GraphInterpolator.__new__(GraphInterpolator)
    itp = GraphInterpolator(
        config=DictConfig(
            {
                "training": {"explicit_times": {"input": [0, 6], "target": [1, 2, 3]}, "multistep_output": 3},
            },
        ),
        graph_data={"data": HeteroData()},
        statistics={},
        statistics_tendencies={},
        data_indices={"data": _make_minimal_index_collection(name_to_index)},
        metadata={},
        supporting_arrays={},
    )
    assert itp.multi_out == 3
    assert itp.boundary_times == [0, 6]
    assert itp.interp_times == [1, 2, 3]
    b, e, g, v = 2, 1, 3, len(name_to_index)
    t = len(itp.imap)
    batch = torch.randn((b, t, e, g, v), dtype=torch.float32)
    loss, metrics, y_preds = itp._step(batch={"data": batch}, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert len(y_preds) == len(itp.interp_times)
    y_pred = torch.stack([pred["data"] for pred in y_preds], dim=1)
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, len(itp.interp_times), e, g, v)


def test_obsinterpolator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: dict[str, IndexCollection],
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.config = config
        self.data_indices = data_indices
        self.model_comm_group = None
        self.grid_shard_shapes = {"data": None}
        self.grid_shard_slice = {"data": None}
        self.model = DummyModel(
            num_output_variables=len(data_indices["data"].model.output),
            output_times=len(config.training.explicit_times.target),
        )
        self.model.pre_processors = {"data": Processors([])}
        self.loss = {"data": DummyLoss()}
        self.dataset_names = list(data_indices.keys())
        self.grid_dim = -2

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    cfg = DictConfig(
        {
            "training": {
                "multistep_input": 6,
                "explicit_times": {"input": [0, 36], "target": [1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30]},
                "known_future_variables": ["U_10M_NWP", "V_10M_NWP", "TD_2M_NWP", "T_2M_NWP", "TOT_PREC_NWP"],
            },
        },
    )
    # Minimal variable space: 2 obs + 5 known-future
    name_to_index = {
        "U_10M_NWP": 0,
        "V_10M_NWP": 1,
        "TD_2M_NWP": 2,
        "T_2M_NWP": 3,
        "TOT_PREC_NWP": 4,
        "OBS_A": 5,
        "OBS_B": 6,
    }
    data_indices = {"data": _make_minimal_index_collection(name_to_index)}

    itp = ObsGraphInterpolator(
        config=cfg,
        graph_data={"data": HeteroData()},
        statistics={},
        statistics_tendencies={},
        data_indices=data_indices,
        metadata={"dataset": {}},
        supporting_arrays={},
    )
    # check known_future_variables mapping and basic timing logic
    assert itp.known_future_variables == [0, 1, 2, 3, 4]
    assert itp.multi_step == 6
    # boundary_times are shifted by multi_step-1 = 5
    assert itp.boundary_times == [5, 41]
    # interp_times shifted by 5
    assert itp.interp_times[0] == 6 and itp.interp_times[1] == 7 and itp.interp_times[-1] == 35
    b, e, g, v = 2, 1, 3, len(name_to_index)  # all variables are prognostic for simplicity
    t = len(itp.imap)  # multi step input
    batch = torch.randn((b, t, e, g, v), dtype=torch.float32)
    loss, metrics, y_pred = itp._step(batch={"data": batch}, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert isinstance(y_pred, torch.Tensor)
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, len(itp.interp_times), e, g, v)


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
        self.multi_step = config.training.multistep_input
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
        self.grid_dim = -2
        self.config = config
        self.loss = {"data": DummyLoss()}
        self.loss_supports_sharding = False
        self.metrics_support_sharding = True
        self.multi_out = 1

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)

    cfg = DictConfig(
        {
            "training": {"multistep_input": 1, "multistep_output": 1},
            "model": {
                "model": {
                    "diffusion": {
                        "rho": 7.0,
                    },
                },
            },
        },
    )

    name_to_index = {"A": 0, "B": 1}
    data_indices = {"data": _make_minimal_index_collection(name_to_index)}

    forecaster = GraphDiffusionForecaster(
        config=cfg,
        graph_data={"data": HeteroData()},
        statistics={"data": {}},
        statistics_tendencies={"data": None},
        data_indices=data_indices,
        metadata={},
        supporting_arrays={},
    )

    b, e, g, v = 2, 1, 4, len(name_to_index)
    t = cfg.training.multistep_input

    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)
    loss, metrics, y_preds = forecaster._step(batch={"data": batch}, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert len(y_preds) == 1
    y_pred = y_preds[0]["data"]
    assert isinstance(y_pred, torch.Tensor)
    assert y_pred.ndim == 5
    assert y_pred.shape == (b, 1, e, g, v)


def test_graphensforecaster_advance_input_handles_time_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for _advance_input with time-aware model outputs.

    GraphEnsForecaster rollouts call _advance_input with y_pred from the model.
    Depending on the model, y_pred may include a time dim (B, T, E, G, V).
    This test ensures the default BaseRolloutGraphModule._advance_input can
    handle that (by taking the last time step) without shape errors.
    """

    def _compute_loss_metrics_stub(
        self: GraphEnsForecaster,
        y_pred: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        step: int | None = None,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        del y, step, validation_mode
        assert isinstance(self, GraphEnsForecaster)
        pred = next(iter(y_pred.values()))
        return torch.zeros(1, device=pred.device, dtype=pred.dtype), {}, y_pred

    monkeypatch.setattr(
        GraphEnsForecaster,
        "compute_loss_metrics",
        _compute_loss_metrics_stub,
        raising=True,
    )

    name_to_index = {"A": 0, "B": 1}
    data_indices = _make_minimal_index_collection(name_to_index)

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)

    forecaster.model = DummyModel(num_output_variables=len(data_indices.model.output), output_times=1)
    forecaster.output_mask = {"data": NoOutputMask()}
    forecaster.loss = {"data": DummyLoss()}
    forecaster.data_indices = {"data": data_indices}
    forecaster.dataset_names = ["data"]
    forecaster.multi_step = 1
    forecaster.multi_out = 1
    forecaster.rollout = 1
    forecaster.nens_per_device = 2
    forecaster.grid_shard_shapes = {"data": None}
    forecaster.grid_shard_slice = {"data": None}
    forecaster.model_comm_group = None
    forecaster.model_comm_group_size = 1
    forecaster.grid_dim = -2

    b, e_dummy, g, v = 2, 1, 4, len(name_to_index)
    batch = {"data": torch.randn((b, forecaster.multi_step + forecaster.rollout, e_dummy, g, v), dtype=torch.float32)}

    # Consume one rollout step and ensure it yields a prediction with a time dim.
    loss, metrics, preds = next(forecaster._rollout_step(batch=batch, rollout=1, validation_mode=False))
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert isinstance(preds, dict)
    assert preds["data"].ndim == 5
    assert preds["data"].shape[0] == b


def test_graphensforecaster_time_dim_does_not_break_advance_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # We only want to validate that the rollout loop can advance the input
    # window when the model returns a time dimension (B, T, E, G, V).
    name_to_index = {"A": 0, "B": 1}
    data_indices = _make_minimal_index_collection(name_to_index)

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.multi_step = 1
    forecaster.multi_out = 1
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
    b, g, v = 2, 4, len(name_to_index)
    # Batch has dummy ensemble dim=1 in the dataset
    batch = {"data": torch.randn((b, forecaster.multi_step + forecaster.rollout, 1, g, v), dtype=torch.float32)}

    # Run one rollout step; should not raise and should yield a 5D prediction
    (loss, metrics, preds) = next(forecaster._rollout_step(batch=batch, rollout=1, validation_mode=False))
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert isinstance(preds, dict)
    assert preds["data"].ndim == 5
    assert preds["data"].shape == (b, 1, forecaster.nens_per_device, g, v)
