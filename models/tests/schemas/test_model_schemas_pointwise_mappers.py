# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.schemas.models import BaseModelSchema


def test_base_model_schema_accepts_pointwise_mapper_configuration():
    schema = BaseModelSchema(
        multistep_input=1,
        num_channels=64,
        keep_batch_sharded=True,
        model={
            "_target_": "anemoi.models.models.AnemoiModelEncProcDec",
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
        bounding=[{"_target_": "anemoi.models.layers.bounding.ReluBounding", "variables": ["tp"]}],
    )

    assert schema.processor.target_ == "anemoi.models.layers.processor.PointWiseMLPProcessor"
    assert schema.processor.dropout_p == 0.0
    assert schema.encoder.target_ == "anemoi.models.layers.mapper.PointWiseForwardMapper"
    assert schema.decoder.target_ == "anemoi.models.layers.mapper.PointWiseBackwardMapper"
