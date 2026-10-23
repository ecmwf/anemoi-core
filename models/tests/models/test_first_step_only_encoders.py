# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Encoders that run at the first rollout step only (``first_step_only``), and their training dropout."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.aggregator import MeanAggregator
from anemoi.models.layers.aggregator import SumAggregator
from anemoi.models.models.base import BaseGraphModel
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec

# --------------------------------------------------------------------------- forward: which latents are aggregated


class _AggregationReached(RuntimeError):
    pass


class _CaptureSum(SumAggregator):
    """The real sum aggregator's type (dropout checks it), recording the latents it is given."""

    def __init__(self) -> None:
        super().__init__(input_channels=4, source_channels={"era5": 4, "std": 4})

    def forward(self, hidden_latent, latents):
        self.latents = dict(latents)
        raise _AggregationReached


class _CaptureMean(MeanAggregator):
    def __init__(self) -> None:
        super().__init__(input_channels=4, source_channels={"era5": 4, "std": 4})

    def forward(self, hidden_latent, latents):
        self.latents = dict(latents)
        raise _AggregationReached


class _GraphProvider(nn.Module):
    def get_edges(self, **kwargs):
        return None, None, None


class _HiddenAttributes(nn.Module):
    def forward(self, node_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(3, 4)


class _Encoder(nn.Module):
    """Returns a latent filled with its own value, so the aggregated sources can be told apart."""

    hidden_dim = 4

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x, **kwargs):
        return x[0], torch.full((3, 4), self.value)


class _StubEnsModel(AnemoiEnsModelEncProcDec):
    """An ERA5 encoder and a first_step_only 'std' encoder, up to the latent aggregation."""

    def __init__(self, aggregator: nn.Module) -> None:
        nn.Module.__init__(self)
        self.input_datasets = ["era5", "std"]
        self.dataset2encoder = {"era5": "era5", "std": "std"}
        self.first_step_only_datasets = {"std"}
        self._graph_name_hidden = "hidden"
        self.input_noise = None
        self.input_noise_dataset = None
        self.node_attributes = _HiddenAttributes()
        self.encoder_graph_provider = nn.ModuleDict({name: _GraphProvider() for name in self.input_datasets})
        self.encoder = nn.ModuleDict({"era5": _Encoder(1.0), "std": _Encoder(2.0)})
        self.latent_aggregator = aggregator

    def _resolve_in_out_sharded(self, dataset_names, grid_shard_sizes):
        return dict.fromkeys(dataset_names, False)

    def _assert_valid_sharding(self, *args, **kwargs) -> None:
        return None

    def _assemble_input(self, x, fcstep, batch_ens_size, grid_shard_sizes=None, model_comm_group=None, **kwargs):
        return torch.ones(3, 4), None, None


def _inputs() -> dict[str, torch.Tensor]:
    # (batch, time, ensemble, grid, variables)
    return {"era5": torch.zeros(1, 2, 1, 3, 2), "std": torch.zeros(1, 2, 1, 3, 2)}


def _aggregated(model: _StubEnsModel, **kwargs) -> dict[str, torch.Tensor]:
    with pytest.raises(_AggregationReached):
        model(_inputs(), **kwargs)
    return model.latent_aggregator.latents


def test_first_step_only_dataset_is_encoded_at_the_first_step() -> None:
    latents = _aggregated(_StubEnsModel(_CaptureSum()), fcstep=0)

    assert list(latents) == ["era5", "std"]
    torch.testing.assert_close(latents["std"], torch.full((3, 4), 2.0))


@pytest.mark.parametrize("fcstep", [1, 2, 39])
def test_first_step_only_dataset_is_left_out_after_the_first_step(fcstep: int) -> None:
    # Even though the input still holds it (as inference and training rollout pass it along).
    latents = _aggregated(_StubEnsModel(_CaptureSum()), fcstep=fcstep)

    assert list(latents) == ["era5"]


def test_dropped_dataset_contributes_a_zero_latent_at_the_first_step() -> None:
    latents = _aggregated(_StubEnsModel(_CaptureSum()), fcstep=0, dropped_input_datasets=("std",))

    torch.testing.assert_close(latents["std"], torch.zeros(3, 4))
    torch.testing.assert_close(latents["era5"], torch.full((3, 4), 1.0))


def test_a_zero_latent_sums_to_the_same_as_leaving_the_dataset_out() -> None:
    aggregator = SumAggregator(input_channels=4, source_channels={"era5": 4, "std": 4})
    era5 = torch.randn(3, 4)
    hidden = torch.zeros(3, 4)

    torch.testing.assert_close(
        aggregator(hidden, {"era5": era5, "std": torch.zeros(3, 4)}), aggregator(hidden, {"era5": era5})
    )


def test_only_first_step_only_datasets_can_be_dropped() -> None:
    with pytest.raises(ValueError, match="Only first_step_only datasets"):
        _StubEnsModel(_CaptureSum())(_inputs(), fcstep=0, dropped_input_datasets=("era5",))


def test_dropping_needs_a_sum_of_the_latents() -> None:
    with pytest.raises(ValueError, match="SumAggregator"):
        _StubEnsModel(_CaptureMean())(_inputs(), fcstep=0, dropped_input_datasets=("std",))


def test_models_pickled_before_the_option_keep_every_encoder() -> None:
    # A model unpickled from an older checkpoint has no instance attribute: the class default applies.
    model = _StubEnsModel(_CaptureSum())
    del model.first_step_only_datasets

    assert list(_aggregated(model, fcstep=3)) == ["era5", "std"]


def test_predict_step_post_processes_only_the_predicted_datasets() -> None:
    # Inference passes every dataset in; the model predicts only the decoded ones.
    class _Processor(nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def forward(self, x, in_place=False):
            return x + self.offset

    def forward(x, **kwargs):
        return {"era5": x["era5"][:, -1:]}

    model = SimpleNamespace(forward=forward, _apply_spatial_preprocessor=BaseGraphModel._apply_spatial_preprocessor)
    processors = nn.ModuleDict({"era5": _Processor(1.0), "std": _Processor(10.0)})
    batch = {"era5": torch.zeros(1, 2, 3, 2), "std": torch.zeros(1, 2, 3, 4)}  # (batch, time, grid, variables)

    y_hat = BaseGraphModel.predict_step(model, batch, processors, processors, n_step_input=2)

    assert list(y_hat) == ["era5"]
    torch.testing.assert_close(y_hat["era5"], torch.full((1, 1, 1, 3, 2), 2.0))


# --------------------------------------------------------------------------- routing checks at construction


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic) + len(self.forcing)


def _indices(prognostic: list[int], forcing: list[int], names: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=prognostic, forcing=forcing),
            output=_IndexGroup(prognostic=prognostic, forcing=[], full=prognostic, diagnostic=[], name_to_index=names),
            _forcing=forcing,
        ),
        data=SimpleNamespace(input=SimpleNamespace(name_to_index=names)),
        name_to_index=names,
    )


def _graph() -> HeteroData:
    graph = HeteroData()
    for name, nodes in (("era5", 2), ("std", 2), ("hidden", 1)):
        graph[name].x = torch.zeros(nodes, 2)
        graph[name].num_nodes = nodes
    return graph


class _RoutingModel(BaseGraphModel):
    supports_first_step_only_encoders = True

    def _build_networks(self, model_config) -> None:
        return None

    def _assemble_input(self, *args, **kwargs):
        raise NotImplementedError

    def _assemble_output(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _NoStepRoutingModel(_RoutingModel):
    supports_first_step_only_encoders = False


def _build(model_cls, *, std_first_step_only=True, decode_std=False):
    mapper = {}
    decoders = {"era5": {"target_datasets": ["era5"], "target_node_features": ["coordinates"], "mapper": mapper}}
    if decode_std:
        decoders["std"] = {"target_datasets": ["std"], "target_node_features": ["coordinates"], "mapper": mapper}
    config = OmegaConf.create(
        {
            "num_channels": 8,
            "node_trainable_parameters": {"era5": 0, "std": 0, "hidden": 0},
            "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
            "encoders": {
                "era5": {
                    "source_datasets": ["era5"],
                    "dataset_fusing_strategy": "not_supported",
                    "first_step_only": False,
                    "mapper": mapper,
                },
                "std": {
                    "source_datasets": ["std"],
                    "dataset_fusing_strategy": "not_supported",
                    "first_step_only": std_first_step_only,
                    "mapper": mapper,
                },
            },
            "decoders": decoders,
            "residual": {
                "datasets": {
                    "era5": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                    "std": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                },
            },
            "bounding": {"datasets": {"era5": [], "std": []}},
        }
    )
    return model_cls(
        model_config=config,
        data_indices={
            "era5": _indices([0], [], {"t": 0}),
            "std": _indices([0] if decode_std else [], [] if decode_std else [0], {"std_t": 0}),
        },
        statistics={"era5": None, "std": None},
        n_step_input=2,
        n_step_output=1,
        graph_data=_graph(),
    )


def test_first_step_only_datasets_are_read_from_the_encoders_config() -> None:
    assert _build(_RoutingModel).first_step_only_datasets == {"std"}
    assert _build(_RoutingModel, std_first_step_only=False).first_step_only_datasets == set()


def test_a_model_without_the_rollout_step_rejects_first_step_only() -> None:
    with pytest.raises(ValueError, match="does not know the rollout step"):
        _build(_NoStepRoutingModel)


def test_a_decoded_dataset_cannot_be_first_step_only() -> None:
    with pytest.raises(ValueError, match="also decoded"):
        _build(_RoutingModel, decode_std=True)
