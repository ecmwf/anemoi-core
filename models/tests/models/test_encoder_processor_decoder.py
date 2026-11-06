# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from anemoi.models.data import TensorLayout
from anemoi.models.layers.aggregator import PointwiseCrossAttentionAggregator
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from tests.batch_builders import build_batch


class _AggregationReached(RuntimeError):
    pass


class _GraphProvider(nn.Module):
    def get_edges(self, **kwargs):
        return torch.zeros(1, 1), torch.zeros(2, 1, dtype=torch.long), None


class _HiddenAttributes(nn.Module):
    def forward(self, node_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(1, 4)


class _SharedEncoder(nn.Module):
    hidden_dim = 4

    def forward(self, x, **kwargs):
        return x[0], x[0]


class _CaptureAggregator(nn.Module):
    def forward(self, hidden_latent: torch.Tensor, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        self.latents = dict(latents)
        raise _AggregationReached


class _SharedEncoderModel(AnemoiModelEncProcDec):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.input_datasets = ["dataset_a", "dataset_b"]
        self.dataset2encoder = {"dataset_a": "shared", "dataset_b": "shared"}
        self.encoder2datasets = {"shared": self.input_datasets}
        self.encoder_fusing_strategy = {"shared": "sequential"}
        self.dataset2hidden = {"dataset_a": "hidden", "dataset_b": "hidden"}
        self.input_dim_latent = 4
        self.node_attributes = _HiddenAttributes()
        self.encoder_graph_provider = nn.ModuleDict(
            {dataset_name: _GraphProvider() for dataset_name in self.input_datasets},
        )
        self.encoder = nn.ModuleDict({"shared": _SharedEncoder()})
        self.encoder_src_projection = nn.ModuleDict()
        self.latent_aggregator = _CaptureAggregator()

    def _hidden_coordinates(self, hidden_name: str | None = None) -> torch.Tensor:
        return torch.zeros(1, 2)

    def _build_networks(self, model_config) -> None:
        raise NotImplementedError

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_sizes=None,
        model_comm_group=None,
        dataset_name: str | None = None,
    ):
        del x, batch_size, grid_shard_sizes, model_comm_group
        value = float(self.input_datasets.index(dataset_name) + 1)
        return torch.zeros(1, 2), torch.full((1, 4), value), None, None, None, None

    def _assemble_output(self, *args, **kwargs):
        raise NotImplementedError


def test_shared_encoder_preserves_each_dataset_latent() -> None:
    model = _SharedEncoderModel()
    inputs = build_batch(
        data={name: torch.zeros(1, 1, 1, 1, 1) for name in model.input_datasets},
        coordinates={name: torch.zeros(1, 2) for name in model.input_datasets},
        layouts={name: TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4) for name in model.input_datasets},
        variables={name: ["a"] for name in model.input_datasets},
        statistics={name: {} for name in model.input_datasets},
    )

    with pytest.raises(_AggregationReached):
        model(inputs, target_forcings=inputs, target_template=inputs.template())

    assert list(model.latent_aggregator.latents) == ["dataset_a", "dataset_b"]
    torch.testing.assert_close(model.latent_aggregator.latents["dataset_a"], torch.full((1, 4), 1.0))
    torch.testing.assert_close(model.latent_aggregator.latents["dataset_b"], torch.full((1, 4), 2.0))


def test_build_latent_aggregator_passes_layer_kernels_to_aggregator() -> None:
    model = _SharedEncoderModel()
    aggregator_config = OmegaConf.create(
        {
            "_target_": "anemoi.models.layers.aggregator.PointwiseCrossAttentionAggregator",
            "num_channels": 8,
            "num_heads": 2,
            "layer_kernels": {
                "Linear": {"_target_": "torch.nn.Linear", "bias": False},
            },
        },
    )

    model._build_latent_aggregator(aggregator_config)

    assert isinstance(model.latent_aggregator, PointwiseCrossAttentionAggregator)
    assert model.latent_aggregator.input_channels == 4
    assert model.latent_aggregator.source_channels == {"dataset_a": 4, "dataset_b": 4}
    assert model.latent_aggregator.hidden_projection.bias is None


# --- Encoder fusion ("none" / "sequential" + fuser) -------------------------------------------

LINEAR_FUSER = {"_target_": "anemoi.models.layers.fuser.LinearFuser", "num_channels": 6}


class _RecordingMapper(nn.Module):
    """Stand-in encoder mapper that records the widths it was built with."""

    def __init__(self, in_channels_src: int, in_channels_dst: int, edge_dim: int, **kwargs) -> None:
        super().__init__()
        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.edge_dim = edge_dim


def _encoder_config(strategy: str, source_datasets: list[str], fuser: dict | None = None):
    return OmegaConf.create(
        {
            "shared": {
                "source_datasets": source_datasets,
                "dataset_fusing_strategy": strategy,
                "fuser": fuser,
                "mapper": {
                    "_target_": "tests.models.test_encoder_processor_decoder._RecordingMapper",
                    "layer_kernels": {"Linear": {"_target_": "torch.nn.Linear", "bias": False}},
                },
            },
        },
    )


def _model_for_encoder_build(strategy: str, in_dims: dict[str, int]) -> _SharedEncoderModel:
    model = _SharedEncoderModel()
    model.input_datasets = list(in_dims)
    model.encoder2datasets = {"shared": list(in_dims)}
    model.encoder_fusing_strategy = {"shared": strategy}
    model.input_dim = dict(in_dims)
    for provider in model.encoder_graph_provider.values():
        provider.edge_dim = 3
    return model


@pytest.mark.parametrize(
    ("strategy", "fuser", "match"),
    [
        ("joint", LINEAR_FUSER, "planned but not yet supported"),
        ("concat", None, "unsupported dataset_fusing_strategy"),
        ("sequential", None, "needs a fuser"),
    ],
)
def test_validate_fusing_strategy_rejects_invalid_settings(strategy: str, fuser: dict | None, match: str) -> None:
    config = _encoder_config(strategy, ["dataset_a"], fuser)["shared"]

    with pytest.raises(ValueError, match=match):
        AnemoiModelEncProcDec._validate_fusing_strategy("shared", config)


def test_validate_fusing_strategy_ignores_fuser_for_none(caplog: pytest.LogCaptureFixture) -> None:
    config = _encoder_config("none", ["dataset_a"], LINEAR_FUSER)["shared"]

    assert AnemoiModelEncProcDec._validate_fusing_strategy("shared", config) == "none"
    assert "fuser setting is ignored" in caplog.text


@pytest.mark.parametrize("in_dims", [{"dataset_a": 3, "dataset_b": 5}, {"dataset_a": 3}])
def test_sequential_builds_linear_fuser_feeding_the_mapper(in_dims: dict[str, int]) -> None:
    from anemoi.models.layers.fuser import LinearFuser

    model = _model_for_encoder_build("sequential", in_dims)

    model._build_encoding_networks(_encoder_config("sequential", list(in_dims), LINEAR_FUSER))

    fuser = model.encoder_src_projection["shared"]
    assert isinstance(fuser, LinearFuser)
    assert model.encoder["shared"].in_channels_src == 6  # fuser.num_channels, also for a single source
    assert model._encoder_projects_sources("shared")
    # The fuser falls back to the mapper's layer_kernels (bias=False here).
    assert all(projection.bias is None for projection in fuser.values())
    # State-dict keys are unchanged from the pre-fuser per-dataset projections: no checkpoint migration.
    assert sorted(k for k in model.state_dict() if k.startswith("encoder_src_projection.")) == sorted(
        f"encoder_src_projection.shared.{d}.weight" for d in in_dims
    )


def test_none_builds_no_fuser_and_feeds_the_assembled_width() -> None:
    model = _model_for_encoder_build("none", {"dataset_a": 4, "dataset_b": 4})

    model._build_encoding_networks(_encoder_config("none", ["dataset_a", "dataset_b"], LINEAR_FUSER))

    assert len(model.encoder_src_projection) == 0
    assert model.encoder["shared"].in_channels_src == 4
    assert not model._encoder_projects_sources("shared")


def test_none_rejects_source_datasets_of_differing_widths() -> None:
    model = _model_for_encoder_build("none", {"dataset_a": 3, "dataset_b": 5})

    with pytest.raises(ValueError, match="same input width"):
        model._build_encoding_networks(_encoder_config("none", ["dataset_a", "dataset_b"]))


def _forward_until_aggregation(model: _SharedEncoderModel, dataset_names: list[str]) -> None:
    inputs = build_batch(
        data={name: torch.zeros(1, 1, 1, 1, 1) for name in dataset_names},
        coordinates={name: torch.zeros(1, 2) for name in dataset_names},
        layouts={name: TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4) for name in dataset_names},
        variables={name: ["a"] for name in dataset_names},
        statistics={name: {} for name in dataset_names},
    )
    model(inputs, target_forcings=inputs, target_template=inputs.template())


def test_none_encodes_the_one_source_dataset_present_in_the_batch() -> None:
    model = _SharedEncoderModel()
    model.encoder_fusing_strategy = {"shared": "none"}

    with pytest.raises(_AggregationReached):
        _forward_until_aggregation(model, ["dataset_b"])

    assert list(model.latent_aggregator.latents) == ["dataset_b"]


def test_none_rejects_several_source_datasets_in_one_batch() -> None:
    model = _SharedEncoderModel()
    model.encoder_fusing_strategy = {"shared": "none"}

    with pytest.raises(ValueError, match=r"batch holds 2 of its source datasets"):
        _forward_until_aggregation(model, ["dataset_a", "dataset_b"])


def test_sequential_projects_each_source_through_its_fuser() -> None:
    from anemoi.models.layers.fuser import LinearFuser

    model = _SharedEncoderModel()
    fuser = LinearFuser(num_channels=4, in_channels={"dataset_a": 4, "dataset_b": 4})
    model.encoder_src_projection = nn.ModuleDict({"shared": fuser})

    with pytest.raises(_AggregationReached):
        _forward_until_aggregation(model, ["dataset_a", "dataset_b"])

    # _SharedEncoder returns its source input as the latent, so the latent is the fuser output.
    with torch.no_grad():
        for value, name in enumerate(["dataset_a", "dataset_b"], start=1):
            expected = fuser(torch.full((1, 4), float(value)), name)
            torch.testing.assert_close(model.latent_aggregator.latents[name], expected)
