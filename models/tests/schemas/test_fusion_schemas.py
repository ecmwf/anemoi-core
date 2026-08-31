# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
from pydantic import TypeAdapter

from anemoi.models.schemas.fusion import CrossAttentionLatentFusionSchema
from anemoi.models.schemas.fusion import LatentFusionSchema
from anemoi.models.schemas.fusion import SumLatentFusionSchema


@pytest.mark.parametrize(
    ("config", "schema_type", "gradient_checkpointing", "attention_implementation"),
    [
        (
            {"_target_": "anemoi.models.layers.fusion.SumLatentFusion"},
            SumLatentFusionSchema,
            False,
            None,
        ),
        (
            {
                "_target_": "anemoi.models.layers.fusion.CrossAttentionLatentFusion",
                "num_heads": 4,
                "dropout_p": 0.1,
            },
            CrossAttentionLatentFusionSchema,
            True,
            "scaled_dot_product_attention",
        ),
    ],
)
def test_latent_fusion_schema(
    config,
    schema_type,
    gradient_checkpointing,
    attention_implementation,
) -> None:
    parsed = TypeAdapter(LatentFusionSchema).validate_python(config)

    assert isinstance(parsed, schema_type)
    assert parsed.gradient_checkpointing is gradient_checkpointing
    if attention_implementation is not None:
        assert parsed.attention_implementation == attention_implementation
