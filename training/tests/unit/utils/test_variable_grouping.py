# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from dataclasses import dataclass

import pytest

from anemoi.training.utils.variables_metadata import ExtractVariableGroupAndLevel
from anemoi.transform.variables import Variable


@dataclass
class MockedVariable:
    param: str
    levtype: str | None = None
    levelist: str | None = None

    def to_variable(self) -> Variable:
        return Variable.from_dict(
            self.param,
            {
                "mars": {
                    "param": self.param,
                    "levtype": self.levtype,
                    "levelist": self.levelist,
                },
            },
        )


@pytest.fixture
def mocked_variable_metadata() -> dict[str, Variable]:
    return {
        "q_100": MockedVariable("q", "pl", "100"),
        "q_200": MockedVariable("q", "pl", "200"),
        "q_500": MockedVariable("q", "pl", "500"),
        "z_500": MockedVariable("z", "pl", "500"),
        "z_ml_500": MockedVariable("z", "ml", "500"),
        "t_500": MockedVariable("t", "pl", "500"),
        "2t": MockedVariable("2t", "sfc", None),
        "tp": MockedVariable("tp", "sfc", None),
    }


SIMPLE_GROUPS = {
    "default": "default",
    "pl": ["q"],
    "sfc": ["tp"],
}

LARGE_GROUPS = {
    "default": "default",
    "pl": ["q", "z"],
    "sfc": ["tp"],
}
COMPLEX_METADATA_LESS_GROUPS = {
    "default": "default",
    "pl": {"param": ["q", "z"]},
}


FILTERED_GROUPS = {
    "default": "default",
    "q": {"param": ["q"]},
    "sfc": {"is_surface_level": True},
    "z_pl": {"is_pressure_level": True, "param": ["z"]},
    "z_ml": {"is_model_level": True, "param": ["z"]},
    "q_500": {"name": ["q_500"]},
}


@pytest.mark.parametrize(
    ("groups", "variable", "expected_group"),
    [
        (SIMPLE_GROUPS, "q_100", "pl"),
        (SIMPLE_GROUPS, "q_500", "pl"),
        (LARGE_GROUPS, "q_500", "pl"),
        (SIMPLE_GROUPS, "z_500", "default"),
        (SIMPLE_GROUPS, "2t", "default"),
        (SIMPLE_GROUPS, "tp", "sfc"),
        # Complex filtered groups
        (FILTERED_GROUPS, "q_500", "q"),
        (FILTERED_GROUPS, "q_100", "q"),
        (FILTERED_GROUPS, "t_500", "default"),
        (FILTERED_GROUPS, "2t", "sfc"),
        (FILTERED_GROUPS, "z_500", "z_pl"),
        (FILTERED_GROUPS, "z_ml_500", "z_ml"),
        (FILTERED_GROUPS, "tp", "sfc"),
        # Complex metadata-less groups
        (COMPLEX_METADATA_LESS_GROUPS, "q_100", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "q_500", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "z_500", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "z_123", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "2t", "default"),
    ],
)
def test_group_matching(
    groups: dict,
    mocked_variable_metadata: dict[str, MockedVariable],
    variable: str,
    expected_group: str,
) -> None:
    """Test that the group matches expected."""
    variable_metadata = {name: value.to_variable() for name, value in mocked_variable_metadata.items()}

    assert ExtractVariableGroupAndLevel(groups, variable_metadata).get_group(variable) == expected_group


@pytest.fixture
def mocked_variable_lacking_metadata() -> dict[str, Variable]:
    return {}


@pytest.mark.parametrize(
    ("groups", "variable", "expected_group"),
    [
        (SIMPLE_GROUPS, "q_100", "pl"),
        (SIMPLE_GROUPS, "q_500", "pl"),
        (LARGE_GROUPS, "q_500", "pl"),
        (SIMPLE_GROUPS, "z_500", "default"),
        (SIMPLE_GROUPS, "2t", "default"),
        (SIMPLE_GROUPS, "tp", "sfc"),
        # Complex metadata-less groups
        (COMPLEX_METADATA_LESS_GROUPS, "q_100", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "q_500", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "z_500", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "z_123", "pl"),
        (COMPLEX_METADATA_LESS_GROUPS, "2t", "default"),
    ],
)
def test_group_matching_without_metadata(
    groups: dict,
    mocked_variable_lacking_metadata: dict[str, MockedVariable],
    variable: str,
    expected_group: str,
) -> None:
    """Test that the group matches the expected without clear metadata."""
    assert ExtractVariableGroupAndLevel(groups, mocked_variable_lacking_metadata).get_group(variable) == expected_group


@pytest.mark.parametrize(
    ("groups", "variable", "expected_group", "error"),
    [
        ({"default": "sfc", "pl": {"is_pressure_level": True}}, "q_100", "pl", ValueError),
        ({"pl": "q_100"}, "q_100", "pl", AssertionError),
    ],
)
def test_group_matching_raises_error(
    groups: dict,
    mocked_variable_lacking_metadata: dict[str, MockedVariable],
    variable: str,
    expected_group: str,
    error: Exception,
) -> None:
    """Test that the group raises an error."""
    with pytest.raises(error):
        assert (
            ExtractVariableGroupAndLevel(groups, mocked_variable_lacking_metadata).get_group(variable) == expected_group
        )


@pytest.mark.parametrize(
    ("variable", "metadata", "expected_level", "expected_variable"),
    [
        # Pressure level variables
        ("q_100", {"param": "q"}, 100, "q"),  # Missing levelist, but variable name has level
        ("q_100", {}, 100, "q"),  # Missing all metadata, but variable name has level
        ("q_100", {"param": "q", "levtype": "sfc", "levelist": "204"}, 100, "q"),  # Incorrect levelist
        ("a_100", {"param": "q", "levtype": "sfc", "levelist": "204"}, 100, "a"),  # Incorrect levelist and param
        (
            "q_100",
            {"param": "q", "levtype": "pl"},
            100,
            "q",
        ),  # If no levelist, and levtype is pl, return level from variable name
        ("q_127", {"param": "q", "levtype": "pl", "levelist": 200}, 200, "q"),  # Trust correctly formatted metadata
        # Surface variables
        ("2t", {"param": "2t"}, None, "2t"),  # Surface var
        ("2t", {"param": "2t", "levtype": "sfc"}, None, "2t"),  # Surface var
        (
            "2t",
            {"param": "2t", "levtype": "sfc", "levelist": 200},
            None,
            "2t",
        ),  # Surface var, but malformed with levelist
    ],
)
def test_failover_to_crack_in_malformed_data(
    variable: str,
    metadata: dict,
    expected_level: int | None,
    expected_variable: str,
) -> None:
    extractor = ExtractVariableGroupAndLevel({"default": "default"}, {variable: {"mars": metadata}})
    level = extractor.get_level(variable)
    assert level == expected_level, f"Expected level {expected_level} for variable {variable}, but got {level}"
    variable_name = extractor.get_param(variable)
    assert (
        variable_name == expected_variable
    ), f"Expected variable name {expected_variable} for {variable}, but got {variable_name}"
