# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.training.utils.variables_metadata import check_loss_variable_units_compatibility
from anemoi.training.utils.variables_metadata import check_variables_metadata_compatibility

# --- Tests for check_loss_variable_units_compatibility ---


def test_check_loss_variable_units_compatible_different_variables() -> None:
    """Test compatible units between different predicted and target variables."""
    variables_metadata = {
        "tp": {"units": "kg m**-2"},
        "imerg": {"units": "kg m**-2"},
    }
    predicted_variables = ["tp"]
    target_variables = ["imerg"]

    # Should not raise
    check_loss_variable_units_compatibility(predicted_variables, target_variables, variables_metadata)


def test_check_loss_variable_units_incompatible_units_raises() -> None:
    """Test that incompatible units between predicted and target raise ValueError."""
    variables_metadata = {
        "tp": {"units": "kg m**-2"},
        "imerg": {"units": "mm"},
    }
    predicted_variables = ["tp"]
    target_variables = ["imerg"]

    with pytest.raises(ValueError, match="Units are not compatible"):
        check_loss_variable_units_compatibility(predicted_variables, target_variables, variables_metadata)


def test_check_loss_variable_units_missing_metadata_warns() -> None:
    """Test that missing variable metadata warns but doesn't error."""
    variables_metadata = {
        "tp": {"units": "kg m**-2"},
        # "imerg" not in metadata
    }
    predicted_variables = ["tp"]
    target_variables = ["imerg"]

    # Should not raise - missing metadata means we can't check
    check_loss_variable_units_compatibility(predicted_variables, target_variables, variables_metadata)


def test_check_loss_variable_units_none_metadata_returns() -> None:
    """Test that None metadata returns without error."""
    # Should not raise
    check_loss_variable_units_compatibility(["tp"], ["imerg"], None)


# --- Tests for check_variables_metadata_compatibility (issue #838 subset) ---


def test_check_variables_metadata_subset_raises_without_allow_subset() -> None:
    """A checkpoint variable absent from the current data raises by default (#838 pre-fix)."""
    ckpt = {"data": {"tp": {"units": "kg m**-2"}, "z_925": {"units": "m**2 s**-2"}}}
    dataset = {"data": {"variables_metadata": {"tp": {"units": "kg m**-2"}}}}  # z_925 dropped
    with pytest.raises(ValueError, match="compatibility check failed"):
        check_variables_metadata_compatibility(ckpt, dataset)


def test_check_variables_metadata_subset_passes_with_allow_subset() -> None:
    """With allow_subset, checkpoint-only variables are ignored and shared ones still check (#838)."""
    ckpt = {"data": {"tp": {"units": "kg m**-2"}, "z_925": {"units": "m**2 s**-2"}}}
    dataset = {"data": {"variables_metadata": {"tp": {"units": "kg m**-2"}}}}  # z_925 dropped
    # Must not raise: z_925 has no counterpart in the current data.
    check_variables_metadata_compatibility(ckpt, dataset, allow_subset=True)


def test_check_variables_metadata_subset_still_checks_shared_units() -> None:
    """allow_subset must not weaken the unit check for the variables that ARE shared."""
    ckpt = {"data": {"tp": {"units": "kg m**-2"}, "z_925": {"units": "m**2 s**-2"}}}
    # tp has incompatible units in the current data; z_925 is dropped.
    dataset = {"data": {"variables_metadata": {"tp": {"units": "mm"}}}}
    with pytest.raises(ValueError, match="compatibility check failed"):
        check_variables_metadata_compatibility(ckpt, dataset, allow_subset=True)
