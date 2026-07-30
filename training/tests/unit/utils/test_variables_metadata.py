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
