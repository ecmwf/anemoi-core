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
from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.aggregator import SumAggregator
from anemoi.models.layers.graph_provider import DynamicGraphProvider
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec


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
            tensor=lambda x, encoded, target, **kwargs: target.coordinates.new_zeros(target.coordinates.shape[0], 4),
        ),
    }
    model.data_indices = {
        "grid": IndexCollection(DictConfig({"forcing": [], "diagnostic": [], "target": []}), {"a": 0}),
    }
    model._build_conditioning_kwargs = lambda *args, **kwargs: ({"grid": {}}, {}, {"grid": {}})
    return model


@pytest.mark.parametrize(
    "model_type", [AnemoiModelEncProcDec, AnemoiTransportModelEncProcDec, AnemoiEnsModelEncProcDec]
)
def test_moving_grids_isolate_samples_and_members(model_type):
    """Real graph construction and model routing keep all four node copies independent."""
    model = _model(model_type)
    layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
    # Different values for each (sample, ensemble member), constant over its two grid points.
    values = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    data = values[:, None, :, None, None].expand(2, 2, 2, 2, 1).clone().requires_grad_()
    batch = Batch(
        data={"grid": data},
        coordinates={"grid": torch.tensor([[[0.0, 0.0], [0.2, 0.2]], [[0.01, 0.01], [0.21, 0.21]]])},
        layouts={"grid": layout},
        variables={"grid": ["a"]},
        statistics={"grid": {}},
    )
    target = batch.with_data({"grid": torch.zeros(2, 1, 2, 2, 1)})

    def forward(inputs):
        if model_type is AnemoiTransportModelEncProcDec:
            return model._forward_transport_network(inputs, target, {"grid": torch.zeros(2, 1, 2, 1, 1)})
        return model(inputs, target)

    output = forward(batch).data["grid"]
    torch.testing.assert_close(output[:, 0, :, 0, 0], values)
    output.sum().backward()
    assert torch.all(data.grad[:, 0] > 0)

    changed_data = data.detach().clone()
    changed_data[0] += 100
    changed = forward(batch.with_data({"grid": changed_data})).data["grid"]
    torch.testing.assert_close(changed[1], output[1])
    torch.testing.assert_close(changed[0], output[0] + 100)


def test_moving_grid_feature_width_uses_time_layout():
    model = _model(AnemoiModelEncProcDec)
    model.is_dataset_static = {"grid": False}
    model.data_layouts = {"grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)}
    model.n_step_input = 3
    model.num_input_channels = {"grid": 2}
    model.dynamic_node_attribute_dims = {}
    # Three timesteps of two variables plus four coordinate features.
    assert model._calculate_input_dim("grid") == 10


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
    inputs = Batch(
        data={"grid": samples},
        coordinates={"grid": coords},
        variables={"grid": ["a"]},
        layouts={"grid": TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True)},
        statistics={"grid": {}},
    )
    if model_type is AnemoiTransportModelEncProcDec:
        target = inputs.with_data({"grid": [torch.zeros_like(sample) for sample in samples]})
        output = model._forward_transport_network(inputs, target, {"grid": torch.zeros(2, 1, 2, 1, 1)})
    else:
        target = inputs.select(variables=[])
        output = model(inputs, target)
    for expected, actual in zip(samples, output.data["grid"], strict=True):
        torch.testing.assert_close(actual, expected)
    sum(sample.sum() for sample in output.data["grid"]).backward()
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
    interface.is_dataset_static = {"grid": True}
    interface.data_layouts = {"grid": TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)}
    interface.n_step_input = 2
    interface.pre_processors = nn.ModuleDict(
        {"grid": Processors([["normalizer", InputNormalizer({"default": "std"})]])}
    )
    interface.post_processors = nn.ModuleDict(
        {"grid": Processors([["normalizer", InputNormalizer({"default": "std"})]], inverse=True)}
    )
    result = interface.predict_step(
        {"grid": torch.full((2, 1, 2, 1), 4.0)}, target={"grid": torch.empty(1, 1, 2, 0)}, gather_out=False
    )["grid"]
    torch.testing.assert_close(result["data"], torch.full((1, 1, 2, 1), 4.0))
    assert result["variables"] == ["a"]
    assert result["layout"] == ("time", "ensemble", "grid", "variables")
    torch.testing.assert_close(result["latitudes"], torch.rad2deg(torch.tensor([0.0, 0.2])))


@pytest.mark.parametrize("members", [1, 2])
def test_sparse_transport_noise_embeddings_follow_member_node_order(members):
    from anemoi.models.data.views import TabularSourceView

    samples = [torch.zeros(members, nodes, 1) for nodes in [2, 3]]
    view = TabularSourceView(
        name="obs",
        data=samples,
        variables=["a"],
        statistics={},
        coordinates=[torch.zeros(nodes, 2) for nodes in [2, 3]],
        layout=TensorLayout(ensemble=0, grid=1, variables=2, time_in_grid=True),
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
