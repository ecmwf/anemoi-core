# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.models.schemas.decoder import GraphTransformerDecoderSchema
from anemoi.models.schemas.encoder import GraphTransformerEncoderSchema


@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_mapper_halo_exchange_config(decoder: bool, enabled: bool) -> None:
    schema = GraphTransformerDecoderSchema if decoder else GraphTransformerEncoderSchema
    config = {
        "_target_": (
            "anemoi.models.layers.mapper.GraphTransformerBackwardMapper"
            if decoder
            else "anemoi.models.layers.mapper.GraphTransformerForwardMapper"
        ),
        "num_channels": 8,
        "num_chunks": 1,
        "num_heads": 2,
        "mlp_hidden_ratio": 2,
        "cpu_offload": False,
        "qk_norm": False,
        "use_halo_exchange": enabled,
    }
    if decoder:
        config["initialise_data_extractor_zero"] = False
    validated = schema.model_validate(config)
    assert validated.model_dump()["use_halo_exchange"] is enabled
