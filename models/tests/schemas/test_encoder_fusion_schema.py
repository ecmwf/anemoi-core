# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Validation of ``encoders.*.dataset_fusing_strategy`` / ``fusion_anchor``."""

import pytest
from pydantic import ValidationError

from anemoi.models.schemas.models import EncodersSchema
from anemoi.models.schemas.models import Model

_MAPPER = {
    "_target_": "anemoi.models.layers.mapper.PointWiseForwardMapper",
    "num_channels": 64,
    "cpu_offload": False,
    "gradient_checkpointing": True,
    "layer_kernels": {},
}


def _encoder(**overrides: object) -> EncodersSchema:
    return EncodersSchema(source_datasets=["hres", "lres"], mapper=_MAPPER, **overrides)


def test_defaults_to_no_fusing_and_no_anchor() -> None:
    encoder = _encoder()

    assert encoder.dataset_fusing_strategy == "not_supported"
    assert encoder.fusion_anchor is None


def test_accepts_fusing_with_an_anchor_from_the_source_datasets() -> None:
    encoder = _encoder(dataset_fusing_strategy="concatenate_inputs_along_variable_dim", fusion_anchor="hres")

    assert encoder.fusion_anchor == "hres"


def test_rejects_anchor_without_a_fusing_strategy() -> None:
    with pytest.raises(ValidationError, match="would be ignored"):
        _encoder(fusion_anchor="hres")


def test_rejects_fusing_without_an_anchor() -> None:
    with pytest.raises(ValidationError, match="requires fusion_anchor"):
        _encoder(dataset_fusing_strategy="concatenate_inputs_along_variable_dim")


def test_rejects_anchor_that_is_not_a_source_dataset() -> None:
    with pytest.raises(ValidationError, match="not in source_datasets"):
        _encoder(dataset_fusing_strategy="concatenate_inputs_along_variable_dim", fusion_anchor="hidden")


def test_rejects_unknown_fusing_strategy() -> None:
    with pytest.raises(ValidationError):
        _encoder(dataset_fusing_strategy="concatenate", fusion_anchor="hres")


@pytest.mark.parametrize(
    "target",
    [
        "anemoi.models.models.AnemoiTransportSpatialDownscalerModelEncProcDec",
        "anemoi.models.models.transport_encoder_processor_decoder." "AnemoiTransportSpatialDownscalerModelEncProcDec",
    ],
)
def test_model_target_enum_accepts_the_spatial_downscaler(target: str) -> None:
    """A model class missing from ``DefinedModels`` makes its whole config unvalidatable."""
    model = Model(**{"_target_": target, "hidden_nodes_name": "hidden"})

    assert model.target_ == target
