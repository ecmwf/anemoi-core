# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.training.schemas.data import DataSchema

_TARGET = "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector"

_MINIMAL_DATA = {
    "format": "zarr",
    "num_features": None,
}


class TestDataSchemaWithSpatialProcessors:
    def test_spatial_processors_absent_by_default(self) -> None:
        schema = DataSchema(**_MINIMAL_DATA)
        assert schema.spatial_processors is None

    def test_spatial_processors_with_edges_name(self) -> None:
        schema = DataSchema(
            **_MINIMAL_DATA,
            spatial_processors={
                "lowres": {
                    "_target_": _TARGET,
                    "edges_name": ["lowres", "to", "hires"],
                },
            },
        )
        assert schema.spatial_processors["lowres"].edges_name == ("lowres", "to", "hires")
