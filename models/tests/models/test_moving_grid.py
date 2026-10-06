# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from anemoi.graphs.edges.attributes import EdgeLength
from anemoi.models.data import GriddedSourceSample
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.aggregator import SumAggregator
from anemoi.models.layers.graph_provider import DynamicGraphProvider
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from tests.batch_builders import build_batch


class _NearestEdges:
    @staticmethod
    def compute_edge_index_from_coords(source_coords, target_coords):
        source = torch.cdist(target_coords, source_coords).argmin(dim=1)
        return torch.stack((source, torch.arange(target_coords.shape[0])))


def _dynamic_graph():
    provider = DynamicGraphProvider.__new__(DynamicGraphProvider)
    nn.Module.__init__(provider)
    provider.edge_builder = _NearestEdges()
    provider.attributes_config = {"length": EdgeLength()}
    provider._edge_dim = 1
    provider._capture_request = None
    provider._captured_graph = None
    return provider


class _NoNodeAttributes:
    num_trainable_parameters = {}

    def __contains__(self, name):
        return False

    def __call__(self, *args, **kwargs):
        return None


class _Encoder(nn.Module):
    def forward(self, x, edge_index, **kwargs):
        source, destination = edge_index
        messages = x[0][source, :1].expand(-1, 4)
        return x[0], x[1].new_zeros(x[1].shape).index_add(0, destination, messages)


class _Decoder(nn.Module):
    def forward(self, x, edge_index, **kwargs):
        source, destination = edge_index
        return x[0].new_zeros(x[1].shape[0], 1).index_add(0, destination, x[0][source, :1])


class _Processor(nn.Module):
    def forward(self, x, **kwargs):
        return x


class _ProcessorGraph:
    def get_edges(self, **kwargs):
        return torch.zeros(0, 1), torch.zeros(2, 0, dtype=torch.long), None


def _model(model_type):
    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model._graph_name_hidden = "hidden"
    model._hidden_coordinates = lambda: torch.tensor([[0.0, 0.0], [0.2, 0.2]])
    model._graph_data = {"hidden": SimpleNamespace(num_nodes=2), "grid": SimpleNamespace(x=model._hidden_coordinates())}
    model.condition_on_residual = False
    model.noise_injector = lambda x, **kwargs: (x, None)
    model.input_datasets = model.target_datasets = ["grid"]
    model.dataset2encoder = model.dataset2decoder = {"grid": "0"}
    model.encoder2datasets = {"0": ["grid"]}
    model.encoder_fusing_strategy = {"0": "none"}
    model.encoder_src_projection = nn.ModuleDict()
    model.encoder = nn.ModuleDict({"0": _Encoder()})
    model.decoder = nn.ModuleDict({"0": _Decoder()})
    model.encoder_graph_provider = nn.ModuleDict({"grid": _dynamic_graph()})
    model.decoder_graph_provider = nn.ModuleDict({"0": _dynamic_graph(), "grid": _dynamic_graph()})
    model.processor_graph_provider = _ProcessorGraph()
    model.processor = _Processor()
    model.latent_aggregator = SumAggregator(input_channels=4, source_channels={"grid": 4})
    model.node_attributes = _NoNodeAttributes()
    model.statistics = {"grid": {}}
    model.residual = {}
    model.latent_skip = False
    model.dynamic_node_attributes = {}
    model.boundings = {"grid": nn.Identity()}
    model.decoders_target_input = {
        "0": SimpleNamespace(
            features=[SimpleNamespace(name="coordinates")],
            tensor=lambda x, encoded, target, target_spec, **kwargs: target_spec.coordinates.new_zeros(
                target_spec.coordinates.shape[0], 4
            ),
        ),
    }
    model.data_indices = {
        "grid": IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), {"a": 0}),
    }
    model._build_conditioning_kwargs = lambda *args, **kwargs: ({"grid": {}}, {}, {"grid": {}})
    return model


@pytest.mark.parametrize(("is_static", "expected"), [(True, 10), (False, 6)])
def test_feature_width_counts_time_steps_per_node(is_static: bool, expected: int):
    """Gridded datasets fold their steps into the features; tabular (non-static) ones carry one step per node."""
    model = _model(AnemoiModelEncProcDec)
    model.is_dataset_static = {"grid": is_static}
    model.n_step_input = {"grid": 3}
    model.num_input_channels = {"grid": 2}
    model.dynamic_node_attribute_dims = {}
    # Gridded: three timesteps of two variables plus four coordinate features.
    # Tabular: the three windows are stacked on the node axis, so one step of two variables plus four.
    assert model._calculate_input_dim("grid") == expected


@pytest.mark.parametrize(
    "model_type", [AnemoiModelEncProcDec, AnemoiEnsModelEncProcDec, AnemoiTransportModelEncProcDec]
)
def test_sparse_ensemble_keeps_sample_and_member_nodes_separate(model_type):
    model = _model(model_type)
    samples = [
        torch.tensor([1.0, 2.0])[:, None, None].expand(2, 2, 1).clone().requires_grad_(),
        torch.tensor([10.0, 20.0])[:, None, None].expand(2, 3, 1).clone().requires_grad_(),
    ]
    coords = [torch.zeros(2, 2), torch.zeros(3, 2)]
    inputs = build_batch(
        data={"grid": samples},
        coordinates={"grid": coords},
        timedeltas={"grid": [torch.zeros(len(c)) for c in coords]},
        boundaries={"grid": [(slice(0, len(c)),) for c in coords]},
        variables={"grid": ["a"]},
        layouts={"grid": TensorLayout(ensemble=0, grid=1, variables=2)},
        statistics={"grid": {}},
    )
    if model_type is AnemoiTransportModelEncProcDec:
        target = inputs.with_data({"grid": [torch.zeros_like(sample) for sample in samples]})
        output = model._forward_transport_network(inputs, target, {"grid": torch.zeros(2, 1, 2, 1, 1)})
    else:
        target = inputs.select(variables=[])
        output = model(inputs, target_forcings=target, target_template=model.output_templates(target))
    for expected, actual in zip(samples, output["grid"].data, strict=True):
        torch.testing.assert_close(actual, expected)
    sum(sample.sum() for sample in output["grid"].data).backward()
    assert all(torch.isfinite(sample.grad).all() and sample.grad.abs().sum() > 0 for sample in samples)


def test_inference_forcing_only_target_preserves_output_metadata():
    from anemoi.models.interface import AnemoiModelInterface
    from anemoi.models.preprocessing import Processors
    from anemoi.models.preprocessing.normalizer import InputNormalizer

    model = _model(AnemoiModelEncProcDec)
    model.statistics = {"grid": {"stdev": torch.tensor([2.0])}}
    interface = AnemoiModelInterface.__new__(AnemoiModelInterface)
    nn.Module.__init__(interface)
    interface.model = model
    interface.data_indices = model.data_indices
    interface.statistics = model.statistics
    interface.statistics_tendencies = None
    interface.is_dataset_static = {"grid": True}
    interface.sample_types = {"grid": GriddedSourceSample}
    interface.n_step_input = {"grid": 2}
    interface.pre_processors = nn.ModuleDict(
        {"grid": Processors([["normalizer", InputNormalizer({"default": "std"})]])}
    )
    interface.post_processors = nn.ModuleDict(
        {"grid": Processors([["normalizer", InputNormalizer({"default": "std"})]], inverse=True)}
    )
    grid = {
        "latitudes": torch.rad2deg(torch.tensor([0.0, 0.2])),
        "longitudes": torch.rad2deg(torch.tensor([0.0, 0.2])),
        "layout": ("time", "ensemble", "grid", "variables"),
    }
    result = interface.predict_step(
        {"grid": {**grid, "data": torch.full((2, 1, 2, 1), 4.0), "variables": ["a"]}},
        target_template={"grid": {**grid, "data": torch.empty(1, 1, 2, 0), "variables": []}},
        gather_out=False,
    )["grid"]
    torch.testing.assert_close(result["data"], torch.full((1, 1, 2, 1), 4.0))
    assert result["variables"] == ["a"]
    assert result["layout"] == ("time", "ensemble", "grid", "variables")
    torch.testing.assert_close(result["latitudes"], torch.rad2deg(torch.tensor([0.0, 0.2])))


@pytest.mark.parametrize("members", [1, 2])
def test_sparse_transport_noise_embeddings_follow_member_node_order(members):

    samples = [torch.zeros(members, nodes, 1) for nodes in [2, 3]]
    view = TabularSource(
        name="obs",
        data=samples,
        variables=["a"],
        statistics={},
        coordinates=[torch.zeros(nodes, 2) for nodes in [2, 3]],
        layout=TensorLayout(ensemble=0, grid=1, variables=2),
        timedeltas=[torch.zeros(nodes) for nodes in [2, 3]],
        boundaries=[(slice(0, nodes),) for nodes in [2, 3]],
    )
    noise = torch.arange(1.0, 2 * members + 1).reshape(2, 1, members, 1, 1).requires_grad_()
    model = _model(AnemoiTransportModelEncProcDec)
    result = model._make_noise_emb_for_view(noise, view)
    expected = [
        float(sample * members + member + 1)
        for sample, nodes in enumerate([2, 3])
        for member in range(members)
        for _ in range(nodes)
    ]
    torch.testing.assert_close(result[:, 0], torch.tensor(expected))
    result.sum().backward()
    torch.testing.assert_close(noise.grad[:, 0, :, 0, 0], torch.tensor([[2.0], [3.0]]).expand(2, members))
