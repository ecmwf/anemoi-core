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

from anemoi.models.variable.variabletokenizer import MeanPoolingTransform
from anemoi.models.variable.variablevocabular import VariableVocabulary

ERA5_METADATA = {
    "2t": {"mars": {"levtype": "sfc", "param": "2t", "time": 0, "step": 0, "date": 19790101}},
    "t_850": {"mars": {"levtype": "pl", "levelist": 850, "param": "t", "time": 0, "step": 0, "date": 19790101}},
    "tp": {"mars": {"levtype": "sfc", "param": "tp", "time": 1800, "step": 6, "date": 19781231}},
    "cos_latitude": {"computed_forcing": True, "constant_in_time": True},
}


def test_from_foundation_shares_param_between_levels():
    vocabulary = VariableVocabulary.from_foundation(ERA5_METADATA)

    tensors = vocabulary.get_variables(["2t", "t_850", "tp", "cos_latitude"])

    assert tensors.param_ids[0] == tensors.param_ids[1] == vocabulary.param_to_id["t"]
    assert tensors.is_forcing.tolist() == [False, False, False, True]
    torch.testing.assert_close(tensors.vertical_levels, torch.tensor([2.0, 850.0, 0.0, 0.0]))


def _metadata(param: str, *, units: str, levtype: str = "sfc") -> dict:
    return {
        "mars": {"param": param, "levtype": levtype, "time": 0, "step": 0},
        "units": units,
    }


def test_pressure_and_precipitation_semantics_follow_issue_1411() -> None:
    metadata = {
        "msl": _metadata("msl", units="Pa"),
        "mslp": _metadata("mslp", units="Pa"),
        "sp": _metadata("sp", units="Pa"),
        "tp": _metadata("tp", units="m"),
    }
    vocabulary = VariableVocabulary.from_foundation(metadata)

    for name in ("msl", "mslp"):
        specification = vocabulary.specification_by_name[name]
        assert specification.param == "p"
        assert specification.vertical_coordinate.type == "height_above_sea"
        assert specification.vertical_coordinate.level == 0.0
        assert specification.value_unit == "Pa"

    surface_pressure = vocabulary.specification_by_name["sp"]
    assert surface_pressure.param == "p"
    assert surface_pressure.vertical_coordinate.type == "height_above_ground"
    assert surface_pressure.vertical_coordinate.level == 0.0

    precipitation = vocabulary.specification_by_name["tp"]
    assert precipitation.param == "tp"
    assert precipitation.vertical_coordinate.type == "height_above_ground"
    assert precipitation.vertical_coordinate.level == 0.0


def test_vocabulary_is_union_of_domain_fields() -> None:
    metadata = {
        "domain_a": {"2t": _metadata("2t", units="K")},
        "domain_b": {"tp": _metadata("tp", units="m")},
    }

    vocabulary = VariableVocabulary.from_datasets(metadata)

    assert set(vocabulary.specification_by_name) == {"2t", "tp"}
    assert vocabulary.specification_by_name["2t"].value_unit == "K"


def test_vocabulary_serialization_preserves_value_units() -> None:
    metadata = {"2t": _metadata("2t", units="K")}
    vocabulary = VariableVocabulary.from_foundation(metadata)

    restored = VariableVocabulary.from_dict(vocabulary.to_dict())

    assert restored.specification_by_name["2t"].value_unit == "K"


def test_tokenizer_has_fixed_width_for_different_domain_field_sets() -> None:
    metadata = {
        "domain_a": {
            "2t": _metadata("2t", units="K"),
            "sp": _metadata("sp", units="Pa"),
        },
        "domain_b": {"tp": _metadata("tp", units="m")},
    }
    tokenizer = MeanPoolingTransform(
        emb_dim=8,
        out_dim=4,
        variable_metadata={name: entry for domain in metadata.values() for name, entry in domain.items()},
    )

    domain_a = tokenizer(torch.ones(2, 3, 1, 5, 2), torch.zeros(10, 0), variables=["2t", "sp"])
    domain_b = tokenizer(torch.ones(2, 3, 1, 5, 1), torch.zeros(10, 0), variables=["tp"])

    assert domain_a.shape == domain_b.shape == (10, 12)


def test_vocabulary_rejects_incompatible_domain_metadata() -> None:
    metadata = {
        "domain_a": {"2t": _metadata("2t", units="K")},
        "domain_b": {"2t": _metadata("2t", units="C")},
    }

    with pytest.raises(ValueError, match="incompatible metadata"):
        VariableVocabulary.from_datasets(metadata)
