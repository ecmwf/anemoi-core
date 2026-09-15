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
from omegaconf import OmegaConf

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


# Observation-style metadata: `param` is the full variable name including the level or channel
# suffix, and everything is declared surface. Such metadata is internally self-consistent, so it
# is trusted by default, yet it is useless for grouping.
OBS_STYLE_METADATA = {
    "z_500": {"mars": {"param": "z_500", "levtype": "sfc"}, "units": "1"},
    "t_850": {"mars": {"param": "t_850", "levtype": "sfc"}},
    "cris_1053": {"mars": {"param": "cris_1053", "levtype": "sfc"}},
    "cris_1109": {"mars": {"param": "cris_1109", "levtype": "sfc"}},
    "2t": {"mars": {"param": "2t", "levtype": "sfc"}},
}


@pytest.mark.parametrize(
    ("variable", "expected_param", "expected_level"),
    [
        ("z_500", "z", 500),
        ("t_850", "t", 850),
        ("cris_1053", "cris", 1053),
        ("2t", "2t", None),
    ],
)
def test_ignore_variables_metadata_cracks_name(
    variable: str,
    expected_param: str,
    expected_level: int | None,
) -> None:
    """With the flag set, self-consistent but useless obs-style metadata is bypassed entirely."""
    extractor = ExtractVariableGroupAndLevel(
        {"default": "sfc", "ignore_variables_metadata": True},
        OBS_STYLE_METADATA,
    )
    assert extractor.get_param(variable) == expected_param
    assert extractor.get_level(variable) == expected_level


def test_metadata_trusted_by_default_for_obs_style_metadata() -> None:
    """Guard the default: without the flag the unhelpful metadata still wins.

    This deliberately asserts the undesirable behaviour, to document that the fix is opt-in.
    """
    extractor = ExtractVariableGroupAndLevel({"default": "sfc"}, OBS_STYLE_METADATA)
    assert extractor.get_param("z_500") == "z_500"
    assert extractor.get_level("z_500") is None


def test_ignore_variables_metadata_restores_group_membership() -> None:
    """The flag also fixes `get_group`, which otherwise matches on the un-cracked param."""
    groups = {"default": "sfc", "ignore_variables_metadata": True, "pl": {"param": ["t", "z"]}}
    extractor = ExtractVariableGroupAndLevel(groups, OBS_STYLE_METADATA)

    assert extractor.get_group("z_500") == "pl"
    assert extractor.get_group("t_850") == "pl"
    assert extractor.get_group("cris_1053") == "sfc"

    # Channels of one instrument collapse to a single (group, param) block
    assert extractor.get_group_and_level("cris_1053")[:2] == ("sfc", "cris")
    assert extractor.get_group_and_level("cris_1109")[:2] == ("sfc", "cris")


def test_ignore_variables_metadata_is_not_treated_as_a_group() -> None:
    """The reserved key must be popped, not iterated as a group specification."""
    extractor = ExtractVariableGroupAndLevel(
        {"default": "sfc", "ignore_variables_metadata": True, "pl": ["z"]},
    )
    assert "ignore_variables_metadata" not in extractor.variable_groups
    assert extractor.ignore_variables_metadata
    assert extractor.get_group("2t") == "sfc"
    assert extractor.get_group("z_500") == "pl"


def test_ignore_variables_metadata_rejects_non_param_group_specs() -> None:
    """Without metadata there is no attribute to match on, so this must fail loudly."""
    extractor = ExtractVariableGroupAndLevel(
        {"default": "sfc", "ignore_variables_metadata": True, "pl": {"is_pressure_level": True}},
        OBS_STYLE_METADATA,
    )
    with pytest.raises(ValueError, match="not found in metadata"):
        extractor.get_group("z_500")


def test_ignore_variables_metadata_from_dictconfig() -> None:
    """Hydra delivers `variable_groups` as a DictConfig, so the pop must follow to_container."""
    groups = OmegaConf.create(
        {"default": "sfc", "ignore_variables_metadata": True, "pl": {"param": ["z"]}},
    )
    extractor = ExtractVariableGroupAndLevel(groups, OBS_STYLE_METADATA)

    assert extractor.ignore_variables_metadata
    assert "ignore_variables_metadata" not in extractor.variable_groups
    assert extractor.get_group_and_level("z_500") == ("pl", "z", 500)


def test_ignore_variables_metadata_constructor_argument() -> None:
    """The flag is also settable programmatically, without the reserved config key."""
    extractor = ExtractVariableGroupAndLevel(
        {"default": "sfc"},
        OBS_STYLE_METADATA,
        ignore_variables_metadata=True,
    )
    assert extractor.get_param("z_500") == "z"
