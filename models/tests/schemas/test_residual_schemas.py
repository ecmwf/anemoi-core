# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import ValidationError

from anemoi.models.schemas.residual import TruncatedConnectionSchema
from anemoi.models.schemas.residual import TruncationConfigDiskSchema
from anemoi.models.schemas.residual import TruncationConfigOnTheFlySchema


def test_truncation_config_disk_valid() -> None:
    TruncationConfigDiskSchema(truncation_up_file_path="up.npz", truncation_down_file_path="down.npz")


def test_truncation_config_on_the_fly_valid_grid() -> None:
    TruncationConfigOnTheFlySchema(grid="o32")


def test_truncation_config_on_the_fly_valid_node_builder() -> None:
    TruncationConfigOnTheFlySchema(
        node_builder={"_target_": "anemoi.graphs.nodes.ReducedGaussianGridNodes", "grid": "o32"}
    )


def test_truncation_config_on_the_fly_requires_grid_or_node_builder() -> None:
    with pytest.raises(ValidationError, match="grid.*node_builder"):
        TruncationConfigOnTheFlySchema()


def test_truncated_connection_on_the_fly_valid() -> None:
    TruncatedConnectionSchema(
        **{
            "_target_": "anemoi.models.layers.residual.TruncatedConnection",
            "truncation_config": {"grid": "o32", "num_nearest_neighbours": 3, "sigma": 1.0},
        }
    )


def test_truncated_connection_disk_valid() -> None:
    TruncatedConnectionSchema(
        **{
            "_target_": "anemoi.models.layers.residual.TruncatedConnection",
            "truncation_config": {
                "truncation_up_file_path": "up.npz",
                "truncation_down_file_path": "down.npz",
            },
        }
    )


def test_truncated_connection_mixed_mode_rejected() -> None:
    """grid (on-the-fly) and file paths (disk) must not coexist in truncation_config."""
    with pytest.raises(ValidationError):
        TruncatedConnectionSchema(
            **{
                "_target_": "anemoi.models.layers.residual.TruncatedConnection",
                "truncation_config": {
                    "grid": "o32",
                    "truncation_up_file_path": "up.npz",
                    "truncation_down_file_path": "down.npz",
                },
            }
        )
