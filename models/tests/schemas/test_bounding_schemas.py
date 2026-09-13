# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.models.schemas.models import Bounding

_CFG = {
    "_target_": "anemoi.models.layers.bounding.HydrostaticGeopotential",
    "levels": [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50],
    "normalizer": {"z": "min-max", "t": "mean-std", "q": "mean-std"},
}


def test_hydrostatic_bounding_schema_in_union() -> None:
    schema = TypeAdapter(Bounding).validate_python(_CFG)
    assert schema.geopotential_units == "m2/s2"
    assert schema.check_finite is False


@pytest.mark.parametrize(
    "override",
    [
        {"levels": [1000, 1000, 500]},
        {"levels": [500]},
        {"normalizer": {"z": "min-max", "t": "mean-std"}},
        {"normalizer": {"z": "cubic", "t": "mean-std", "q": "mean-std"}},
        {"geopotential_units": "km"},
    ],
)
def test_hydrostatic_bounding_schema_rejects(override: dict) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Bounding).validate_python({**_CFG, **override})
