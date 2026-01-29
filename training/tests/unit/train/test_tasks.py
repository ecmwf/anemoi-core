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


@pytest.mark.parametrize(
    ("multi_step", "multi_out", "expected"),
    [
        (2, 3, [2.0, 3.0]),
        (2, 2, [1.0, 2.0]),
        (3, 2, [-1.0, 1.0, 2.0]),
    ],
)
def test_rollout_advance_input_keeps_latest_steps(
    multi_step: int,
    multi_out: int,
    expected: list[float],
) -> None:
    name_to_index = {"A": 0, "B": 1}
    data_indices = _make_minimal_index_collection(name_to_index)

    forecaster = GraphEnsForecaster.__new__(GraphEnsForecaster)
    pl.LightningModule.__init__(forecaster)
    forecaster.multi_step = multi_step
    forecaster.multi_out = multi_out
    forecaster.output_mask = {"data": NoOutputMask()}
    forecaster.data_indices = {"data": data_indices}
    forecaster.grid_shard_slice = {"data": None}

    b, e, g, v = 1, 1, 2, len(name_to_index)
    x = torch.full((b, forecaster.multi_step, e, g, v), -1.0, dtype=torch.float32)
    y_pred = torch.stack(
        [torch.full((b, e, g, v), float(step), dtype=torch.float32) for step in range(1, forecaster.multi_out + 1)],
        dim=1,
    )
    batch = torch.zeros((b, forecaster.multi_step + forecaster.multi_out, e, g, v), dtype=torch.float32)

    updated = forecaster._advance_dataset_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        dataset_name="data",
    )
    for idx, value in enumerate(expected):
        assert torch.all(updated[:, idx] == value)
