# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
import torch

from anemoi.models.data_indices.index import DataIndex
from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.data_indices.tensor import OutputTensorIndex


@pytest.fixture()
def fake_data():
    name_to_index = {"x": 0, "y": 1, "z": 2, "q": 3, "other": 4}
    forcing = ["x", "y"]
    diagnostic = ["z"]
    prognostic = ["q", "other"]
    return forcing, prognostic, diagnostic, name_to_index


@pytest.fixture()
def input_tensor_index(fake_data):
    forcing, prognostic, diagnostic, name_to_index = fake_data
    return InputTensorIndex(
        includes=forcing + prognostic,
        prognostic=prognostic,
        forcing=forcing,
        diagnostic=diagnostic,
        name_to_index=name_to_index,
    )


@pytest.fixture()
def output_tensor_index(fake_data):
    forcing, prognostic, diagnostic, name_to_index = fake_data
    return OutputTensorIndex(
        includes=prognostic + diagnostic,
        prognostic=prognostic,
        forcing=forcing,
        diagnostic=diagnostic,
        name_to_index=name_to_index,
    )


def test_dataindex_init(fake_data, input_tensor_index, output_tensor_index) -> None:
    forcing, _, diagnostic, name_to_index = fake_data
    data_index = DataIndex(forcing=forcing, diagnostic=diagnostic, name_to_index=name_to_index)
    assert data_index.input == input_tensor_index
    assert data_index.output == output_tensor_index


def test_output_tensor_index_full(output_tensor_index) -> None:
    expected_output = torch.Tensor([2, 3, 4]).to(torch.int)
    assert torch.allclose(output_tensor_index.full, expected_output)


def test_output_tensor_index_only(output_tensor_index) -> None:
    expected_output = torch.Tensor([2]).to(torch.int)
    assert torch.allclose(output_tensor_index.diagnostic, expected_output)


def test_output_tensor_index_prognostic(output_tensor_index) -> None:
    expected_output = torch.Tensor([3, 4]).to(torch.int)
    assert torch.allclose(output_tensor_index.prognostic, expected_output)


def test_output_tensor_index_todict(output_tensor_index) -> None:
    expected_output = {
        "full": torch.Tensor([2, 3, 4]).to(torch.int),
        "target": torch.Tensor([]).to(torch.int),
        "diagnostic": torch.Tensor([2]).to(torch.int),
        "forcing": torch.Tensor([0, 1]).to(torch.int),
        "prognostic": torch.Tensor([3, 4]).to(torch.int),
        "name_to_index": dict(x=0, y=1, z=2, q=3, other=4),
    }
    for key, value in output_tensor_index.todict().items():
        assert key in expected_output
        if isinstance(value, dict):
            assert value == expected_output[key]
        else:
            assert torch.allclose(value, expected_output[key])


def test_output_tensor_index_getattr(output_tensor_index) -> None:
    assert output_tensor_index.full is not None
    with pytest.raises(AttributeError):
        output_tensor_index.z


def test_output_tensor_index_name_position_mapping(output_tensor_index) -> None:
    assert output_tensor_index.ordered_names == ["z", "q", "other"]
    assert output_tensor_index.name_to_position == {"z": 0, "q": 1, "other": 2}
    assert output_tensor_index.positions_for_names(["q", "other"]) == [1, 2]


def test_output_tensor_index_positions_for_names_missing(output_tensor_index) -> None:
    with pytest.raises(ValueError, match="not available in this tensor index-space"):
        output_tensor_index.positions_for_names(["x"])


def test_input_tensor_index_full(input_tensor_index) -> None:
    expected_output = torch.Tensor([0, 1, 3, 4]).to(torch.int)
    assert torch.allclose(input_tensor_index.full, expected_output)


def test_input_tensor_index_only(input_tensor_index) -> None:
    expected_output = torch.Tensor([0, 1]).to(torch.int)
    assert torch.allclose(input_tensor_index.forcing, expected_output)


def test_input_tensor_index_prognostic(input_tensor_index) -> None:
    expected_output = torch.Tensor([3, 4]).to(torch.int)
    assert torch.allclose(input_tensor_index.prognostic, expected_output)


def test_input_tensor_index_todict(input_tensor_index) -> None:
    expected_output = {
        "full": torch.Tensor([0, 1, 3, 4]).to(torch.int),
        "target": torch.Tensor([]).to(torch.int),
        "diagnostic": torch.Tensor([2]).to(torch.int),
        "forcing": torch.Tensor([0, 1]).to(torch.int),
        "prognostic": torch.Tensor([3, 4]).to(torch.int),
        "name_to_index": dict(x=0, y=1, z=2, q=3, other=4),
    }
    for key, value in input_tensor_index.todict().items():
        assert key in expected_output
        if isinstance(value, dict):
            assert value == expected_output[key]
        else:
            assert torch.allclose(value, expected_output[key])


def test_input_tensor_index_getattr(input_tensor_index) -> None:
    assert input_tensor_index.full is not None
    with pytest.raises(AttributeError):
        input_tensor_index.z


def test_input_tensor_index_name_position_mapping(input_tensor_index) -> None:
    assert input_tensor_index.ordered_names == ["x", "y", "q", "other"]
    assert input_tensor_index.name_to_position == {"x": 0, "y": 1, "q": 2, "other": 3}
    assert input_tensor_index.positions_for_names(["x", "other"]) == [0, 3]
