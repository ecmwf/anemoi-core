# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytest
from pydantic import ValidationError

from anemoi.models.schemas.models import EncodersSchema
from anemoi.models.schemas.models import LinearNodeEmbedderSchema

MAPPER = {
    "_target_": "anemoi.models.layers.mapper.PointWiseForwardMapper",
    "num_channels": 64,
    "cpu_offload": False,
    "gradient_checkpointing": True,
    "layer_kernels": {},
}
LINEAR_NODE_EMBEDDER = {"_target_": "anemoi.models.layers.node_embedder.LinearNodeEmbedder", "num_channels": 512}


def _encoder(**kwargs) -> dict:
    return {"source_datasets": ["era5"], "mapper": MAPPER, **kwargs}


def test_none_is_the_default_and_needs_no_node_embedder() -> None:
    schema = EncodersSchema(**_encoder())

    assert schema.dataset_fusing_strategy == "none"
    assert schema.node_embedder is None


def test_sequential_accepts_a_linear_node_embedder() -> None:
    schema = EncodersSchema(
        **_encoder(
            source_datasets=["era5", "cerra"], dataset_fusing_strategy="sequential", node_embedder=LINEAR_NODE_EMBEDDER
        )
    )

    assert isinstance(schema.node_embedder, LinearNodeEmbedderSchema)
    assert schema.node_embedder.num_channels == 512


def test_sequential_requires_a_node_embedder() -> None:
    with pytest.raises(ValidationError, match="needs a node embedder"):
        EncodersSchema(**_encoder(dataset_fusing_strategy="sequential"))


@pytest.mark.parametrize("num_channels", [0, -1])
def test_node_embedder_num_channels_must_be_positive(num_channels: int) -> None:
    with pytest.raises(ValidationError):
        EncodersSchema(
            **_encoder(
                dataset_fusing_strategy="sequential",
                node_embedder={**LINEAR_NODE_EMBEDDER, "num_channels": num_channels},
            )
        )


def test_node_embedder_target_must_be_known() -> None:
    with pytest.raises(ValidationError):
        EncodersSchema(
            **_encoder(
                dataset_fusing_strategy="sequential",
                node_embedder={**LINEAR_NODE_EMBEDDER, "_target_": "torch.nn.Linear"},
            )
        )


def test_joint_is_rejected_as_not_yet_supported() -> None:
    with pytest.raises(ValueError, match="'joint' is planned but not yet supported"):
        EncodersSchema(**_encoder(dataset_fusing_strategy="joint", node_embedder=LINEAR_NODE_EMBEDDER))


def test_none_ignores_a_node_embedder_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        schema = EncodersSchema(**_encoder(node_embedder=LINEAR_NODE_EMBEDDER))

    assert schema.dataset_fusing_strategy == "none"
    assert "node embedder is ignored" in caplog.text


def test_fusion_projection_dim_is_no_longer_accepted() -> None:
    with pytest.raises(ValidationError, match="fusion_projection_dim"):
        EncodersSchema(
            **_encoder(
                dataset_fusing_strategy="sequential", node_embedder=LINEAR_NODE_EMBEDDER, fusion_projection_dim=8
            )
        )
