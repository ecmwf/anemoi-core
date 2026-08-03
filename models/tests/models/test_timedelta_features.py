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

from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.data.views import TabularSourceView
from anemoi.models.models.base import split_graph_config
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict


def _model_with_timedelta_attributes(dtype: str = "float32") -> AnemoiModelEncProcDec:
    model = AnemoiModelEncProcDec.__new__(AnemoiModelEncProcDec)
    torch.nn.Module.__init__(model)
    model.dynamic_node_attributes = {}
    model.dynamic_node_attribute_dims = {}
    model._configure_dynamic_node_attributes(
        DotDict(
            {
                "obs": {
                    "attributes": {
                        "timedeltas": {
                            "_target_": "anemoi.graphs.nodes.attributes.Timedeltas",
                            "scale_seconds": 3600.0,
                            "periods": [24.0],
                            "dtype": dtype,
                        },
                    },
                },
            },
        ),
    )
    return model


def test_forecaster_encodes_configured_timedelta_node_features() -> None:
    model = _model_with_timedelta_attributes()
    flat_view = SimpleNamespace(
        timedeltas=torch.tensor([-3600.0, 0.0, 21600.0]),
        coordinates=torch.zeros(3, 2),
    )

    actual = model._encode_dynamic_node_attributes("obs", flat_view)

    assert model.dynamic_node_attribute_dims["obs"] == 3
    assert actual is not None
    assert actual.shape == (3, 3)
    assert torch.allclose(actual[:, 0], torch.tensor([-1.0, 0.0, 6.0]))


def test_forecaster_requires_timedeltas_when_node_encoding_is_configured() -> None:
    model = _model_with_timedelta_attributes()
    flat_view = SimpleNamespace(timedeltas=None, coordinates=torch.zeros(3, 2))

    with pytest.raises(ValueError, match="does not provide timedeltas"):
        model._encode_dynamic_node_attributes("obs", flat_view)


def test_forecaster_input_dimensions_include_configured_timedelta_features() -> None:
    model = _model_with_timedelta_attributes()
    model.is_dataset_static = {"obs": False}
    model.num_input_channels = {"obs": 5}
    model.num_input_channels_decoding_forcings = {"obs": 2}
    model.node_attributes = SimpleNamespace(num_trainable_parameters={"obs": 4})
    model.use_encoder_data_output = {"obs": False}

    assert model._calculate_input_dim("obs") == 16
    assert model._calculate_target_dim("obs") == 13


def test_forecaster_assembles_timedelta_features_for_both_mappers() -> None:
    model = _model_with_timedelta_attributes()
    model.residual = {}
    model.node_attributes = {}
    model.use_encoder_data_output = {"obs": False}
    view = TabularSourceView(
        name="obs",
        data=[torch.ones(3, 1)],
        variables=["value"],
        statistics={},
        coordinates=[torch.zeros(3, 2)],
        timedeltas=[torch.tensor([-3600.0, 0.0, 3600.0])],
        layout=TensorLayout(grid=0, variables=1, time_in_grid=True),
    )

    input_coords, input_features, _, _, batch_sizes, input_timedeltas = model._assemble_input(
        view,
        batch_size=1,
        dataset_name="obs",
    )
    target_coords, target_features, _, target_batch_sizes, target_timedeltas = model._assemble_target(
        view,
        encoder_data_output=None,
        batch_size=1,
        dataset_name="obs",
    )

    assert input_features.shape == (3, 8)
    assert target_features.shape == (3, 8)
    assert torch.equal(input_coords, target_coords)
    assert torch.equal(input_timedeltas, view.timedeltas[0])
    assert torch.equal(target_timedeltas, view.timedeltas[0])
    assert batch_sizes == target_batch_sizes == (3,)


def test_forecaster_casts_configured_node_dtype_to_mapper_input_dtype() -> None:
    model = _model_with_timedelta_attributes(dtype="float64")
    model.residual = {}
    model.node_attributes = {}
    view = TabularSourceView(
        name="obs",
        data=[torch.ones(2, 1, dtype=torch.float32)],
        variables=["value"],
        statistics={},
        coordinates=[torch.zeros(2, 2)],
        timedeltas=[torch.tensor([0.0, 3600.0])],
        layout=TensorLayout(grid=0, variables=1, time_in_grid=True),
    )

    _, input_features, _, _, _, _ = model._assemble_input(view, batch_size=1, dataset_name="obs")

    assert input_features.dtype == torch.float32


def test_forecaster_reuses_encoder_output_without_duplicate_node_features() -> None:
    model = _model_with_timedelta_attributes()
    model.node_attributes = {}
    model.use_encoder_data_output = {"obs": True}
    view = TabularSourceView(
        name="obs",
        data=[torch.ones(2, 1)],
        variables=["value"],
        statistics={},
        coordinates=[torch.zeros(2, 2)],
        timedeltas=[torch.tensor([0.0, 3600.0])],
        layout=TensorLayout(grid=0, variables=1, time_in_grid=True),
    )
    encoder_output = torch.ones(2, 8)

    _, target_features, _, _, target_timedeltas = model._assemble_target(
        view,
        encoder_data_output=encoder_output,
        batch_size=1,
        dataset_name="obs",
    )

    assert target_features is encoder_output
    assert torch.equal(target_timedeltas, view.timedeltas[0])


def test_split_graph_config_retains_dynamic_timedelta_attributes() -> None:
    graph_config = DotDict(
        {
            "nodes": {
                "obs": {
                    "attributes": {
                        "timedeltas": {
                            "_target_": "anemoi.graphs.nodes.attributes.Timedeltas",
                            "periods": [24.0],
                        },
                    },
                },
                "hidden": {"node_builder": {}},
            },
            "edges": [],
        },
    )

    static_config, dynamic_config = split_graph_config(
        graph_config,
        is_dataset_static={"obs": False},
        hidden_nodes_name="hidden",
    )

    assert "hidden" in static_config.nodes
    assert dynamic_config.nodes.obs.attributes.timedeltas.periods == [24.0]
