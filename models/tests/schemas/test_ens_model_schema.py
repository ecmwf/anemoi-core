# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.schemas.models import BaseModelSchema
from anemoi.models.schemas.models import EnsModelSchema


_BASE_SCHEMA_KWARGS = dict(
    num_channels=64,
    keep_batch_sharded=True,
    model={
        "_target_": "anemoi.models.models.AnemoiEnsModelEncProcDec",
        "hidden_nodes_name": "data",
        "latent_skip": True,
    },
    processor={
        "_target_": "anemoi.models.layers.processor.PointWiseMLPProcessor",
        "num_layers": 2,
        "num_chunks": 1,
        "mlp_hidden_ratio": 4,
        "cpu_offload": False,
        "gradient_checkpointing": True,
        "layer_kernels": {},
    },
    encoder={
        "_target_": "anemoi.models.layers.mapper.PointWiseForwardMapper",
        "cpu_offload": False,
        "gradient_checkpointing": True,
        "layer_kernels": {},
    },
    decoder={
        "_target_": "anemoi.models.layers.mapper.PointWiseBackwardMapper",
        "initialise_data_extractor_zero": False,
        "cpu_offload": False,
        "gradient_checkpointing": True,
        "layer_kernels": {},
    },
    trainable_parameters={"data": 0, "hidden": 0},
    residual={"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1},
    output_mask={"_target_": "anemoi.training.utils.masks.NoOutputMask"},
    bounding=[],
)

_NOISE_INJECTOR = {
    "_target_": "anemoi.models.layers.ensemble.NoOpNoiseInjector",
}


def test_ens_model_schema_allows_extra_keys() -> None:
    """Test that EnsModelSchema allows extra keys and does not quietly drop them.

    Explicitly tests for the issue in ecmwf/anemoi-core#889 where PydanticBaseModel
    was quietly dropping keys required in ens_encoder_processor_decoder.
    """
    schema = EnsModelSchema(
        **_BASE_SCHEMA_KWARGS,
        noise_injector=_NOISE_INJECTOR,
        condition_on_residual=False,
    )
    assert schema.noise_injector.target_ == "anemoi.models.layers.ensemble.NoOpNoiseInjector"
    assert schema.condition_on_residual is False

    model_dump = schema.model_dump(by_alias=True)
    assert model_dump["noise_injector"]["_target_"] == "anemoi.models.layers.ensemble.NoOpNoiseInjector"
    assert model_dump["condition_on_residual"] is False


def test_base_model_schema_allows_extra_keys() -> None:
    """Test that BaseModelSchema allows and preserves extra keys.

    Explicitly tests for the issue in ecmwf/anemoi-core#889 where PydanticBaseModel
    was quietly dropping keys required in ens_encoder_processor_decoder.
    """
    schema = BaseModelSchema(
        **_BASE_SCHEMA_KWARGS,
        noise_injector=_NOISE_INJECTOR,
        condition_on_residual=False,
    )

    model_dump = schema.model_dump(by_alias=True)
    assert model_dump["noise_injector"] == _NOISE_INJECTOR
    assert model_dump["condition_on_residual"] is False
