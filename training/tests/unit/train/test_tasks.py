from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.training.train.tasks.obsinterpolator import ObsGraphInterpolator


class DummyLoss(torch.nn.Module):
    """Minimal loss used via torch.utils.checkpoint in _step."""

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_pred.squeeze() - y) ** 2)


class SimpleObsModel:
    """Shape-aware minimal model used in _step test."""

    def __init__(self, num_output_vars: int):
        self.num_output_vars = num_output_vars

    def pre_processors(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __call__(
        self,
        x: torch.Tensor,
        model_comm_group: Any | None = None,
        grid_shard_shapes: Any | None = None,
    ) -> torch.Tensor:
        self.called_with = {
            "x_shape": tuple(x.shape),
            "model_comm_group": model_comm_group,
            "grid_shard_shapes": grid_shard_shapes,
        }
        b, _, e, g, _ = x.shape
        return torch.zeros((b, 1, e, g, self.num_output_vars), dtype=x.dtype, device=x.device)


def _make_minimal_index_collection(name_to_index: dict[str, int]) -> IndexCollection:
    cfg = DictConfig({"data": {"forcing": [], "diagnostic": []}})
    return IndexCollection(config=cfg, name_to_index=name_to_index)


class DummyModel:
    """Minimal stub for AnemoiModelInterface used by the task."""

    def __init__(self) -> None:
        self.called_with: dict[str, Any] | None = None

    def __call__(
        self,
        x: torch.Tensor,
        model_comm_group: Any | None = None,
        grid_shard_shapes: Any | None = None,
    ) -> torch.Tensor:
        # Record call for assertions
        self.called_with = {
            "x_shape": tuple(x.shape),
            "model_comm_group": model_comm_group,
            "grid_shard_shapes": grid_shard_shapes,
        }
        return x + 1


def test_obsinterpolator_forward_pass_minimal() -> None:
    # Build a minimal task instance without running BaseGraphModule.__init__
    task = ObsGraphInterpolator.__new__(ObsGraphInterpolator)
    dummy_model = DummyModel()
    task.model = dummy_model
    task.model_comm_group = "dummy_comm_group"
    task.grid_shard_shapes = {"dummy": True}
    x = torch.zeros((2, 1, 1, 4, 3), dtype=torch.float32)
    y = task.forward(x)

    # forward preserve shape
    assert y.shape == x.shape
    assert torch.allclose(y, x + 1)

    # check the model was called with the expected kwargs
    assert dummy_model.called_with is not None
    assert dummy_model.called_with["x_shape"] == tuple(x.shape)
    assert dummy_model.called_with["model_comm_group"] == task.model_comm_group
    assert dummy_model.called_with["grid_shard_shapes"] == task.grid_shard_shapes


def test_obsinterpolator_init_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch BaseGraphModule.__init__ to avoid heavy initialisation
    def _stub_bgm_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        del graph_data, truncation_data, statistics, statistics_tendencies, metadata, supporting_arrays
        # ensure Module internals are ready before assigning nn.Modules
        pl.LightningModule.__init__(self)
        self.config = config
        self.data_indices = data_indices
        self.model_comm_group = None
        self.grid_shard_shapes = None
        self.model = SimpleObsModel(num_output_vars=len(data_indices.data.input.name_to_index))

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_bgm_init, raising=True)
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
        truncation_data={},
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


def test_obsinterpolator_step_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_bgm_init(
        self: BaseGraphModule,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        del graph_data, truncation_data, statistics, statistics_tendencies, metadata, supporting_arrays
        pl.LightningModule.__init__(self)
        self.config = config
        self.data_indices = data_indices
        self.model_comm_group = None
        self.grid_shard_shapes = None
        self.model = SimpleObsModel(num_output_vars=len(data_indices.data.input.name_to_index))
        self.loss = DummyLoss()

    monkeypatch.setattr(BaseGraphModule, "__init__", _stub_bgm_init, raising=True)

    cfg = DictConfig(
        {
            "training": {
                "multistep_input": 6,
                "explicit_times": {"input": [0, 36], "target": [1, 2, 3]},
                "known_future_variables": ["U_10M_NWP", "V_10M_NWP"],
            },
        },
    )
    name_to_index = {
        "U_10M_NWP": 0,
        "V_10M_NWP": 1,
        "OBS_X": 2,
        "OBS_Y": 3,
        "OBS_Z": 4,
    }
    data_indices = _make_minimal_index_collection(name_to_index)

    itp = ObsGraphInterpolator(
        config=cfg,
        graph_data=HeteroData(),
        truncation_data={},
        statistics={},
        statistics_tendencies={},
        data_indices=data_indices,
        metadata={"dataset": {}},
        supporting_arrays={},
    )
    b, e, g, v = 2, 1, 3, len(name_to_index)
    t = len(itp.imap)  # multi step input
    batch = torch.randn((b, t, e, g, v), dtype=torch.float32)

    loss, metrics, y_preds = itp._step(batch=batch, batch_idx=0, validation_mode=False)

    assert isinstance(loss, torch.Tensor)
    assert metrics == {}
    # y_pred is extended per batch element per interpolation step
    assert len(y_preds) == b * len(itp.interp_times)
