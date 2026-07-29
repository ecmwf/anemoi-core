# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.utils import check_loss_tree_variable_units


def test_check_loss_tree_recurses_into_combined_loss() -> None:
    """Test that the tree walker finds variable mappers inside CombinedLoss."""
    data_indices = IndexCollection(
        DictConfig({"forcing": ["forcing"], "diagnostic": [], "target": ["imerg"]}),
        {"tp": 0, "forcing": 1, "imerg": 2},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MAELoss",
                        "predicted_variables": ["tp"],
                        "target_variables": ["imerg"],
                    },
                ],
            },
        ),
        data_indices=data_indices,
    )

    # Compatible units - should not raise
    variables_metadata = {"tp": {"units": "kg m**-2"}, "imerg": {"units": "kg m**-2"}}
    check_loss_tree_variable_units(loss, variables_metadata)

    # Incompatible units - should raise
    variables_metadata_bad = {"tp": {"units": "kg m**-2"}, "imerg": {"units": "mm"}}
    with pytest.raises(ValueError, match="Units are not compatible"):
        check_loss_tree_variable_units(loss, variables_metadata_bad)
