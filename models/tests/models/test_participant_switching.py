# (C) Copyright 2026- Anemoi contributors.
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
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import ParticipantSwitchingGraphProvider
from anemoi.models.models.base import BaseGraphModel

EDGE_ATTRIBUTES = ["edge_length"]


class ParticipantDummyModel(BaseGraphModel):
    """Minimal model building the three edge providers of an encoder-processor-decoder."""

    def _build_networks(self, model_config) -> None:
        trainable_size = model_config.get("test_trainable_size", 0)

        self.encoder_graph_provider = torch.nn.ModuleDict()
        self.decoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            self.encoder_graph_provider[dataset_name] = self._create_graph_provider(
                src_name=dataset_name,
                dst_name=self._graph_name_hidden,
                edge_attributes=EDGE_ATTRIBUTES,
                trainable_size=trainable_size,
                trainable_size_key="model.encoders.enc.mapper.trainable_size",
            )
            self.decoder_graph_provider[dataset_name] = self._create_graph_provider(
                src_name=self._graph_name_hidden,
                dst_name=dataset_name,
                edge_attributes=EDGE_ATTRIBUTES,
                trainable_size=0,
                trainable_size_key="model.decoders.dec.mapper.trainable_size",
            )

        self.processor_graph_provider = self._create_graph_provider(
            src_name=self._graph_name_hidden,
            dst_name=self._graph_name_hidden,
            edge_attributes=EDGE_ATTRIBUTES,
            trainable_size=0,
            trainable_size_key="model.processor.trainable_size",
        )

    def _assemble_input(self, x, batch_size, grid_shard_sizes=None, model_comm_group=None):
        return x

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        return x_out

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices() -> dict:
    dataset_indices = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0], forcing=[]),
            output=_IndexGroup(prognostic=[0], full=[0], diagnostic=[], name_to_index={"var": 0}),
            _forcing=[],
        ),
        data=SimpleNamespace(input=SimpleNamespace(name_to_index={"var": 0})),
        name_to_index={"var": 0},
    )
    return {"data": dataset_indices}


def _add_edges(graph: HeteroData, src: str, dst: str, num_src: int, num_dst: int) -> None:
    src_index, dst_index = torch.meshgrid(torch.arange(num_src), torch.arange(num_dst), indexing="ij")
    edge_index = torch.stack([src_index.flatten(), dst_index.flatten()])
    graph[(src, "to", dst)].edge_index = edge_index
    graph[(src, "to", dst)].edge_length = torch.ones(edge_index.shape[1], 1)


def _add_participant(graph: HeteroData, participant: str | None, num_data_nodes: int) -> None:
    data_name = "data" if participant is None else f"data_{participant}"
    hidden_name = "hidden" if participant is None else f"hidden_{participant}"

    graph[data_name].x = torch.full((num_data_nodes, 2), float(num_data_nodes))
    graph[data_name].num_nodes = num_data_nodes
    graph[hidden_name].x = torch.zeros(2, 2)
    graph[hidden_name].num_nodes = 2

    _add_edges(graph, data_name, hidden_name, num_data_nodes, 2)
    _add_edges(graph, hidden_name, hidden_name, 2, 2)
    _add_edges(graph, hidden_name, data_name, 2, num_data_nodes)


def _participant_graph(num_data_nodes: dict[str, int]) -> HeteroData:
    graph = HeteroData()
    for participant, num_nodes in num_data_nodes.items():
        _add_participant(graph, participant, num_nodes)
    return graph


def _single_domain_graph(num_data_nodes: int = 3) -> HeteroData:
    graph = HeteroData()
    _add_participant(graph, None, num_data_nodes)
    return graph


def _model_config(trainable_parameters: dict[str, int] | None = None, trainable_size: int = 0) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "node_trainable_parameters": trainable_parameters or {"data": 0, "hidden": 0},
                "test_trainable_size": trainable_size,
                "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
                "encoders": {
                    "enc": {
                        "source_datasets": ["data"],
                        "dataset_fusing_strategy": "not_supported",
                        "mapper": {},
                    },
                },
                "decoders": {
                    "dec": {
                        "target_datasets": ["data"],
                        "target_node_features": ["coordinates"],
                        "mapper": {},
                    },
                },
                "residual": {"datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection"}}},
                "bounding": {"datasets": {"data": []}},
            },
        },
    )


def _build_model(graph: HeteroData, **config_kwargs) -> ParticipantDummyModel:
    return ParticipantDummyModel(
        model_config=_model_config(**config_kwargs),
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph,
    )


def test_model_builds_one_provider_per_participant() -> None:
    model = _build_model(_participant_graph({"west": 3, "north": 5}))

    assert model.participants == ["west", "north"]
    assert model.active_participant == "west"

    encoder = model.encoder_graph_provider["data"]
    assert isinstance(encoder, ParticipantSwitchingGraphProvider)
    assert encoder.participants == ["west", "north"]
    assert isinstance(model.processor_graph_provider, ParticipantSwitchingGraphProvider)


def test_set_active_participant_switches_edges_and_node_attributes() -> None:
    model = _build_model(_participant_graph({"west": 3, "north": 5}))

    west_edges = model.encoder_graph_provider["data"].active.edge_attr.shape[0]
    west_coords = model.node_attributes.sin_cos_coordinates("data")

    model.set_active_participant("north")

    assert model.active_participant == "north"
    assert model.node_attributes.active_participant == "north"
    assert model.encoder_graph_provider["data"].active_participant == "north"
    assert model.decoder_graph_provider["data"].active_participant == "north"
    assert model.processor_graph_provider.active_participant == "north"

    north_edges = model.encoder_graph_provider["data"].active.edge_attr.shape[0]
    assert west_edges == 3 * 2
    assert north_edges == 5 * 2

    north_coords = model.node_attributes.sin_cos_coordinates("data")
    assert north_coords.shape[0] == 5
    assert west_coords.shape[0] == 3


def test_participant_sub_providers_are_registered_as_submodules() -> None:
    model = _build_model(_participant_graph({"west": 3, "north": 5}))

    buffer_names = {name for name, _ in model.named_buffers()}
    assert any(name.startswith("encoder_graph_provider.data.providers.west.") for name in buffer_names)
    assert any(name.startswith("encoder_graph_provider.data.providers.north.") for name in buffer_names)
    assert "node_attributes.latlons_data_west" in buffer_names
    assert "node_attributes.latlons_data_north" in buffer_names


def test_single_domain_graph_is_unaffected() -> None:
    model = _build_model(_single_domain_graph())

    assert model.participants == []
    assert model.active_participant is None
    assert not isinstance(model.encoder_graph_provider["data"], ParticipantSwitchingGraphProvider)
    assert model.node_attributes.num_nodes == {"data": 3, "hidden": 2}

    # Callers need not know whether the participants share a graph.
    model.set_active_participant("west")
    assert model.active_participant is None


def test_unknown_participant_is_rejected() -> None:
    model = _build_model(_participant_graph({"west": 3, "north": 5}))

    with pytest.raises(ValueError, match="Unknown participant 'south'"):
        model.set_active_participant("south")


def test_inconsistent_participants_are_rejected() -> None:
    graph = _participant_graph({"west": 3, "north": 5})
    del graph["hidden_north"]

    with pytest.raises(ValueError, match="must have the same participants"):
        _build_model(graph)


def test_trainable_edges_are_rejected_for_multiple_participants() -> None:
    with pytest.raises(ValueError, match="model.encoders.enc.mapper.trainable_size must be 0"):
        _build_model(_participant_graph({"west": 3, "north": 5}), trainable_size=2)


def test_trainable_node_attributes_are_rejected_for_multiple_participants() -> None:
    with pytest.raises(ValueError, match="model.node_trainable_parameters.data must be 0"):
        _build_model(
            _participant_graph({"west": 3, "north": 5}),
            trainable_parameters={"data": 4, "hidden": 0},
        )


def test_trainable_node_attributes_are_allowed_for_a_single_domain() -> None:
    model = _build_model(_single_domain_graph(), trainable_parameters={"data": 4, "hidden": 0})

    assert model.node_attributes.attr_ndims["data"] == 2 * 2 + 4
    assert model.node_attributes.get_tensor("data").trainable is not None
