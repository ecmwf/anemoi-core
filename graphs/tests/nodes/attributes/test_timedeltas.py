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

from anemoi.graphs.nodes.attributes import Timedeltas
from anemoi.graphs.schemas.node_attributes_schemas import NodeAttributeSchemas


def test_timedeltas_scalar_and_fourier_values():
    encoder = Timedeltas(scale_seconds=3600.0, periods=[24.0], dtype="float64")
    timedeltas = torch.tensor([-6, 0, 6, 12, 24], dtype=torch.float64) * 3600

    actual = encoder.compute(timedeltas)
    scaled = torch.tensor([-6, 0, 6, 12, 24], dtype=torch.float64)
    phase = 2 * torch.pi * scaled / 24
    expected = torch.stack((scaled, torch.sin(phase), torch.cos(phase)), dim=-1)

    assert encoder.ndim == 3
    assert actual.dtype == torch.float64
    assert torch.allclose(actual, expected, atol=1e-12)


def test_timedeltas_empty_periods_are_scalar_only():
    encoder = Timedeltas(periods=[])

    actual = encoder.compute(torch.tensor([-3600.0, 0.0, 3600.0]))

    assert encoder.ndim == 1
    assert actual.shape == (3, 1)
    assert torch.equal(actual, torch.tensor([[-1.0], [0.0], [1.0]]))


@pytest.mark.parametrize("scale_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_timedeltas_reject_invalid_scale(scale_seconds):
    with pytest.raises(ValueError, match="scale_seconds"):
        Timedeltas(scale_seconds=scale_seconds)


@pytest.mark.parametrize("period", [0.0, -1.0, float("inf"), float("nan")])
def test_timedeltas_reject_invalid_period(period):
    with pytest.raises(ValueError, match="periods"):
        Timedeltas(periods=[period])


def test_timedeltas_reject_invalid_dtype():
    with pytest.raises(ValueError, match="dtype"):
        Timedeltas(dtype="int64")


def test_timedeltas_reject_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        Timedeltas().compute(torch.ones(3, 2))


def test_timedeltas_fail_clearly_during_static_graph_creation():
    graph = HeteroData()
    graph["nodes"].x = torch.zeros(2, 2)

    with pytest.raises(RuntimeError, match="runtime-only"):
        Timedeltas().compute(graph, "nodes")


def test_timedeltas_node_schema_validation():
    adapter = TypeAdapter(NodeAttributeSchemas)
    model = adapter.validate_python(
        {
            "_target_": "anemoi.graphs.nodes.attributes.Timedeltas",
            "scale_seconds": 60.0,
            "periods": [60.0, 1440.0],
            "dtype": "float64",
        }
    )

    assert model.scale_seconds == 60.0
    assert model.periods == [60.0, 1440.0]


@pytest.mark.parametrize(
    "config",
    [
        {"_target_": "anemoi.graphs.nodes.attributes.Timedeltas", "scale_seconds": 0.0},
        {"_target_": "anemoi.graphs.nodes.attributes.Timedeltas", "periods": [-1.0]},
    ],
)
def test_timedeltas_node_schema_rejects_invalid_settings(config):
    with pytest.raises(ValidationError):
        TypeAdapter(NodeAttributeSchemas).validate_python(config)
