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

    def __call__(
        self,
        x: torch.Tensor,
        model_comm_group: Any | None = None,
        grid_shard_slice: Any | None = None,
        grid_shard_shapes: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
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


class DummyDiffusionModel(DummyModel):

    def __init__(self, num_output_variables: int | None = None) -> None:
        super().__init__(num_output_variables=num_output_variables, output_times=1)
        self.sigma_max = 4.0
        self.sigma_min = 1.0
        self.sigma_data = 0.5

    def fwd_with_preconditioning(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # behave like diffusion: call forward and combine
        assert sigma.shape[0] == x.shape[0]
        assert all(sigma.shape[i] == 1 for i in range(1, sigma.ndim))
        pred = self(x, **kwargs)
        if y_noised.ndim == 4:
            y_noised = y_noised.unsqueeze(1)
        return y_noised + 0.1 * pred


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    cfg = DictConfig({"data": {"forcing": [], "diagnostic": []}})
    return IndexCollection(config=cfg, name_to_index=name_to_index)


def test_graphinterpolator(monkeypatch: pytest.MonkeyPatch) -> None:
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
        self.model = DummyModel(output_times=len(config.training.explicit_times.target), add_skip=True)
        self.model_comm_group = None
        self.grid_shard_slice = None
        self.grid_shard_shapes = None
        self.is_first_step = False  # avoid scaler init
        self.updating_scalars = {}
        self.data_indices = data_indices
        self.config = config
        self.loss = DummyLoss()

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)
    name_to_index = {"A": 0, "B": 1}
    itp = GraphInterpolator.__new__(GraphInterpolator)
    itp = GraphInterpolator(
        config=DictConfig({"training": {"explicit_times": {"input": [0, 6], "target": [1, 2, 3]}}}),
        graph_data=HeteroData(),
        statistics={},
        statistics_tendencies={},
        data_indices=_make_minimal_index_collection(name_to_index),
        metadata={},
        supporting_arrays={},
    )
    assert itp.multi_out == 3
    assert itp.boundary_times == [0, 6]
    assert itp.interp_times == [1, 2, 3]
    b, e, g, v = 2, 1, 3, len(name_to_index)
    t = len(itp.imap)
    batch = torch.randn((b, t, e, g, v), dtype=torch.float32)
    loss, metrics, y_pred = itp._step(batch=batch, validation_mode=False)
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
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
        data_indices: IndexCollection,
        metadata: dict | None = None,
        supporting_arrays: dict | None = None,
    ) -> None:
        del graph_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.config = config
        self.data_indices = data_indices
        self.model_comm_group = None
        self.grid_shard_shapes = None
        self.grid_shard_slice = None
        self.model = DummyModel(
            num_output_variables=len(data_indices.model.output),
            output_times=len(config.training.explicit_times.target),
        )
        self.loss = DummyLoss()

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
    data_indices = _make_minimal_index_collection(name_to_index)

    itp = ObsGraphInterpolator(
        config=cfg,
        graph_data=HeteroData(),
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
    loss, metrics, y_pred = itp._step(batch=batch, validation_mode=False)

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
        model = DummyDiffusionModel(num_output_variables=len(data_indices.model.output))
        self.model = DummyDiffusion(model=model)
        self.model_comm_group = None
        self.grid_shard_shapes = None
        self.grid_shard_slice = None
        self.is_first_step = False
        self.updating_scalars = {}
        self.data_indices = data_indices
        self.config = config
        self.loss = DummyLoss()
        self.loss_supports_sharding = False

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_init, raising=True)

    cfg = DictConfig(
        {
            "training": {"multistep_input": 1},
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
    data_indices = _make_minimal_index_collection(name_to_index)

    forecaster = GraphDiffusionForecaster(
        config=cfg,
        graph_data=HeteroData(),
        statistics={},
        statistics_tendencies={},
        data_indices=data_indices,
        metadata={},
        supporting_arrays={},
    )

    b, e, g, v = 2, 1, 4, len(name_to_index)
    t = cfg.training.multistep_input

    batch = torch.randn((b, t + 1, e, g, v), dtype=torch.float32)
    loss, metrics, y_pred = forecaster._step(batch=batch, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
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
        y_pred: torch.Tensor,
        y: torch.Tensor,
        step: int | None = None,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        del y, step, validation_mode
        assert isinstance(self, GraphEnsForecaster)
        return torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype), {}, y_pred

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
    forecaster.output_mask = NoOutputMask()
    forecaster.loss = DummyLoss()
    forecaster.data_indices = data_indices
    forecaster.multi_step = 1
    forecaster.rollout = 1
    forecaster.nens_per_device = 2
    forecaster.grid_shard_shapes = None
    forecaster.grid_shard_slice = None
    forecaster.model_comm_group = None

    b, e_dummy, g, v = 2, 1, 4, len(name_to_index)
    batch = torch.randn((b, forecaster.multi_step + forecaster.rollout, e_dummy, g, v), dtype=torch.float32)

    # Consume one rollout step and ensure it yields a prediction with a time dim.
    loss, metrics, preds = next(forecaster._rollout_step(batch=batch, rollout=1, validation_mode=False))
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert isinstance(preds, torch.Tensor)
    assert preds.ndim == 5
    assert preds.shape[0] == b


def test_graphensforecaster_time_dim_does_not_break_advance_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # We only want to validate that the rollout loop can advance the input
    # window when the model returns a time dimension (B, T, E, G, V).
    name_to_index = {"A": 0, "B": 1}
    data_indices = _make_minimal_index_collection(name_to_index)

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.multi_step = 1
    forecaster.rollout = 1
    forecaster.nens_per_device = 2
    forecaster.model = DummyModel(num_output_variables=len(data_indices.model.output), output_times=1)
    forecaster.model_comm_group = None
    forecaster.grid_shard_shapes = None
    forecaster.grid_shard_slice = None
    forecaster.output_mask = NoOutputMask()
    forecaster.data_indices = data_indices

    def _compute_loss_metrics(
        y_pred: torch.Tensor,
        y: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        del y, args, kwargs
        return torch.zeros(1, dtype=y_pred.dtype, device=y_pred.device), {}, y_pred

    monkeypatch.setattr(forecaster, "compute_loss_metrics", _compute_loss_metrics)
    b, g, v = 2, 4, len(name_to_index)
    # Batch has dummy ensemble dim=1 in the dataset
    batch = torch.randn((b, forecaster.multi_step + forecaster.rollout, 1, g, v), dtype=torch.float32)

    # Run one rollout step; should not raise and should yield a 5D prediction
    (loss, metrics, preds) = next(forecaster._rollout_step(batch=batch, rollout=1, validation_mode=False))
    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    assert isinstance(preds, torch.Tensor)
    assert preds.ndim == 5
