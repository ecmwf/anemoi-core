# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import ValidationError

from anemoi.models.schemas.spatial_processors import CrossGridProjectorSchema

_TARGET = "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector"


class TestCrossGridProjectorSchema:
    def test_valid_with_file_path(self):
        schema = CrossGridProjectorSchema(**{"_target_": _TARGET, "file_path": "/some/matrix.npz"})
        assert schema.target_ == _TARGET
        assert schema.file_path == "/some/matrix.npz"
        assert schema.edges_name is None
        assert schema.row_normalize is True
        assert schema.autocast is False

    def test_valid_with_edges_name(self):
        schema = CrossGridProjectorSchema(
            **{
                "_target_": _TARGET,
                "edges_name": ("lowres", "to", "hires"),
                "edge_weight_attribute": "weight",
            }
        )
        assert schema.edges_name == ("lowres", "to", "hires")
        assert schema.edge_weight_attribute == "weight"
        assert schema.file_path is None

    def test_error_when_neither_source_provided(self):
        with pytest.raises(ValidationError, match="exactly one of 'file_path' or 'edges_name'"):
            CrossGridProjectorSchema(**{"_target_": _TARGET})

    def test_error_when_both_sources_provided(self):
        with pytest.raises(ValidationError, match="exactly one of 'file_path' or 'edges_name'"):
            CrossGridProjectorSchema(
                **{
                    "_target_": _TARGET,
                    "file_path": "/some/matrix.npz",
                    "edges_name": ("lowres", "to", "hires"),
                }
            )

    def test_error_on_wrong_target(self):
        with pytest.raises(ValidationError):
            CrossGridProjectorSchema(
                **{"_target_": "anemoi.models.preprocessing.some_other.Class", "file_path": "/m.npz"}
            )

    def test_error_on_edges_name_wrong_length(self):
        with pytest.raises(ValidationError):
            CrossGridProjectorSchema(**{"_target_": _TARGET, "edges_name": ("only_two", "items")})
