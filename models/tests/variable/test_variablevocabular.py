# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy

import pytest
import torch

from anemoi.models.variable.variablevocabular import VariableVocabulary
from anemoi.models.variable.variablevocabular import merge_variables_metadata

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


def test_merge_keeps_union_of_datasets():
    danra = {"2t": copy.deepcopy(ERA5_METADATA["2t"]), "sp": {"mars": {"levtype": "sfc", "param": "sp"}}}
    danra["2t"]["mars"]["date"] = 19900901

    merged = merge_variables_metadata({"era5": ERA5_METADATA, "danra": danra})

    assert set(merged) == {"2t", "t_850", "tp", "cos_latitude", "sp"}


def test_merge_rejects_same_name_for_different_variables():
    danra = {"t_850": {"mars": {"levtype": "pl", "levelist": 500, "param": "t", "time": 0, "step": 0}}}

    with pytest.raises(ValueError, match="t_850"):
        merge_variables_metadata({"era5": ERA5_METADATA, "danra": danra})
