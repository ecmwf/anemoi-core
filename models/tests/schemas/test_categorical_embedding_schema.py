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

from anemoi.models.schemas.models import CategoricalEmbeddingSchema


def test_defaults() -> None:
    schema = CategoricalEmbeddingSchema(codes=[49001, 21009])
    assert schema.embedding_dim == 8
    assert schema.unknown_prob == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"codes": []},
        {"codes": [0, 1]},
        {"codes": [5, 5]},
        {"codes": [1.5]},
        {"codes": [1], "unknown_prob": 1.0},
        {"codes": [1], "embedding_dim": 0},
        {"codes": [1], "extra": True},
    ],
)
def test_invalid(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        CategoricalEmbeddingSchema(**kwargs)
