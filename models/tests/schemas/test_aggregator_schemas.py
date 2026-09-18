# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import TypeAdapter

from anemoi.models.schemas.aggregator import AggregatorSchema
from anemoi.models.schemas.aggregator import ConcatAggregatorSchema
from anemoi.models.schemas.aggregator import CrossAttentionAggregatorSchema
from anemoi.models.schemas.aggregator import MeanAggregatorSchema
from anemoi.models.schemas.aggregator import SumAggregatorSchema


@pytest.mark.parametrize(
    ("config", "schema_type"),
    [
        ({"_target_": "anemoi.models.layers.aggregator.SumAggregator"}, SumAggregatorSchema),
        ({"_target_": "anemoi.models.layers.aggregator.MeanAggregator"}, MeanAggregatorSchema),
        ({"_target_": "anemoi.models.layers.aggregator.ConcatAggregator"}, ConcatAggregatorSchema),
        (
            {
                "_target_": "anemoi.models.layers.aggregator.CrossAttentionAggregator",
                "num_channels": 64,
                "num_heads": 4,
            },
            CrossAttentionAggregatorSchema,
        ),
    ],
)
def test_aggregator_schema(config: dict, schema_type: type) -> None:
    parsed = TypeAdapter(AggregatorSchema).validate_python(config)

    assert isinstance(parsed, schema_type)


def test_cross_attention_aggregator_schema_defaults() -> None:
    parsed = TypeAdapter(AggregatorSchema).validate_python(
        {
            "_target_": "anemoi.models.layers.aggregator.CrossAttentionAggregator",
            "num_channels": 64,
            "num_heads": 4,
        },
    )

    assert parsed.attention_implementation == "scaled_dot_product_attention"
    assert parsed.gradient_checkpointing is True
