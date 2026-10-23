# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from pydantic import TypeAdapter
from pydantic import ValidationError
from torch_geometric.data import HeteroData

from anemoi.graphs.edges.attributes import Timedeltas
from anemoi.graphs.schemas.edge_attributes_schemas import EdgeAttributeSchema

EDGE_TYPE = ("source", "to", "target")


@pytest.fixture
def timedelta_graph():
    graph = HeteroData()
    graph["source"].x = torch.zeros(3, 2)
    graph["source"].timedeltas = torch.tensor([-3600.0, 0.0, 3600.0])
    graph["target"].x = torch.zeros(3, 2)
    graph["target"].timedeltas = torch.tensor([7200.0, -7200.0, 0.0])
    graph[EDGE_TYPE].edge_index = torch.tensor([[0, 1, 2], [2, 0, 1]])
    return graph


@pytest.mark.parametrize(
    ("node_axis", "expected"),
    [
        ("source", [-1.0, 0.0, 1.0]),
        ("target", [0.0, 2.0, -2.0]),
    ],
)
def test_timedeltas_propagate_selected_endpoint(timedelta_graph, node_axis, expected):
    builder = Timedeltas(node_axis=node_axis)

    actual = builder(
        x=(timedelta_graph["source"], timedelta_graph["target"]),
        edge_index=timedelta_graph[EDGE_TYPE].edge_index,
    )

    assert builder.ndim == 1
    assert actual.shape == (3, 1)
    assert torch.equal(actual, torch.tensor(expected, device=actual.device).unsqueeze(-1))


@pytest.mark.parametrize("node_axis", ["source", "target"])
def test_timedeltas_require_selected_endpoint_attribute(timedelta_graph, node_axis):
    del timedelta_graph[node_axis].timedeltas
    builder = Timedeltas(node_axis=node_axis)

    with pytest.raises(ValueError, match=f"selected {node_axis} nodes"):
        builder(
            x=(timedelta_graph["source"], timedelta_graph["target"]),
            edge_index=timedelta_graph[EDGE_TYPE].edge_index,
        )


@pytest.mark.parametrize(
    "timedeltas",
    [
        torch.ones(3, 2),
        torch.ones(2),
    ],
)
def test_timedeltas_reject_shape_errors(timedelta_graph, timedeltas):
    timedelta_graph["source"].timedeltas = timedeltas
    builder = Timedeltas(node_axis="source")

    with pytest.raises(ValueError, match="shape|values"):
        builder(
            x=(timedelta_graph["source"], timedelta_graph["target"]),
            edge_index=timedelta_graph[EDGE_TYPE].edge_index,
        )


@pytest.mark.parametrize("node_axis", ["invalid", "", None])
def test_timedeltas_reject_invalid_node_axis(node_axis):
    with pytest.raises(ValueError, match="node_axis"):
        Timedeltas(node_axis=node_axis)


@pytest.mark.parametrize("scale_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_timedeltas_reject_invalid_scale(scale_seconds):
    with pytest.raises(ValueError, match="scale_seconds"):
        Timedeltas(node_axis="source", scale_seconds=scale_seconds)


def test_timedeltas_edge_schema_validation():
    adapter = TypeAdapter(EdgeAttributeSchema)
    model = adapter.validate_python(
        {
            "_target_": "anemoi.graphs.edges.attributes.Timedeltas",
            "node_axis": "target",
            "scale_seconds": 60.0,
            "dtype": "float64",
        }
    )

    assert model.node_axis == "target"
    assert model.scale_seconds == 60.0


@pytest.mark.parametrize(
    "config",
    [
        {
            "_target_": "anemoi.graphs.edges.attributes.Timedeltas",
            "node_axis": "invalid",
        },
        {
            "_target_": "anemoi.graphs.edges.attributes.Timedeltas",
            "node_axis": "source",
            "scale_seconds": 0.0,
        },
    ],
)
def test_timedeltas_edge_schema_rejects_invalid_settings(config):
    with pytest.raises(ValidationError):
        TypeAdapter(EdgeAttributeSchema).validate_python(config)
