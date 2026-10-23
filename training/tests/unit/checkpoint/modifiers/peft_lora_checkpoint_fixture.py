# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A PEFT adapter checkpoint written once with peft 0.18.1, as a compatibility fixture.

``RECORD`` is the adapter record exactly as it was stored in the checkpoint metadata, and
``STATE_DICT_SHAPES`` the state-dict keys and shapes of ``TrainingModule`` in
``test_peft.py`` with that record applied. Both are what a peft upgrade could break; the
weight values are not, so the test fills them in. Do not regenerate this file: it pins
what old checkpoints look like.
"""

# fmt: off
RECORD = {"adapters": {"default": {"config": {"alora_invocation_tokens": None,
                                     "alpha_pattern": {},
                                     "arrow_config": None,
                                     "auto_mapping": None,
                                     "base_model_name_or_path": None,
                                     "bias": "none",
                                     "corda_config": None,
                                     "ensure_weight_tying": False,
                                     "eva_config": None,
                                     "exclude_modules": None,
                                     "fan_in_fan_out": False,
                                     "inference_mode": False,
                                     "init_lora_weights": True,
                                     "layer_replication": None,
                                     "layers_pattern": None,
                                     "layers_to_transform": None,
                                     "loftq_config": {},
                                     "lora_alpha": 8,
                                     "lora_bias": False,
                                     "lora_dropout": 0.0,
                                     "megatron_config": None,
                                     "megatron_core": "megatron.core",
                                     "modules_to_save": None,
                                     "peft_type": "LORA",
                                     "peft_version": "0.18.1",
                                     "qalora_group_size": 16,
                                     "r": 4,
                                     "rank_pattern": {},
                                     "revision": None,
                                     "target_modules": "processor\\.0\\.lin_query|processor\\.0\\.lin_value|processor\\.0\\.mlp\\.0|processor\\.0\\.mlp\\.2|processor\\.1\\.lin_query|processor\\.1\\.lin_value|processor\\.1\\.mlp\\.0|processor\\.1\\.mlp\\.2",  # noqa: E501
                                     "target_parameters": None,
                                     "task_type": None,
                                     "trainable_token_indices": None,
                                     "use_dora": False,
                                     "use_qalora": False,
                                     "use_rslora": False},
                          "type": "lora"}},
 "lineage": [],
 "merged": False,
 "modules": ["model.model.processor.0.lin_query",
             "model.model.processor.0.lin_value",
             "model.model.processor.0.mlp.0",
             "model.model.processor.0.mlp.2",
             "model.model.processor.1.lin_query",
             "model.model.processor.1.lin_value",
             "model.model.processor.1.mlp.0",
             "model.model.processor.1.mlp.2"],
 "trainable": ["model.model.decoder.data.bias", "model.model.decoder.data.weight"],
 "version": 1}

STATE_DICT_SHAPES = {"model.model.decoder.data.bias": [4],
 "model.model.decoder.data.weight": [4, 8],
 "model.model.encoder.data.bias": [8],
 "model.model.encoder.data.weight": [8, 4],
 "model.model.processor.0.lin_query.base_layer.bias": [8],
 "model.model.processor.0.lin_query.base_layer.weight": [8, 8],
 "model.model.processor.0.lin_query.lora_A.default.weight": [4, 8],
 "model.model.processor.0.lin_query.lora_B.default.weight": [8, 4],
 "model.model.processor.0.lin_value.base_layer.bias": [8],
 "model.model.processor.0.lin_value.base_layer.weight": [8, 8],
 "model.model.processor.0.lin_value.lora_A.default.weight": [4, 8],
 "model.model.processor.0.lin_value.lora_B.default.weight": [8, 4],
 "model.model.processor.0.mlp.0.base_layer.bias": [16],
 "model.model.processor.0.mlp.0.base_layer.weight": [16, 8],
 "model.model.processor.0.mlp.0.lora_A.default.weight": [4, 8],
 "model.model.processor.0.mlp.0.lora_B.default.weight": [16, 4],
 "model.model.processor.0.mlp.2.base_layer.bias": [8],
 "model.model.processor.0.mlp.2.base_layer.weight": [8, 16],
 "model.model.processor.0.mlp.2.lora_A.default.weight": [4, 16],
 "model.model.processor.0.mlp.2.lora_B.default.weight": [8, 4],
 "model.model.processor.0.norm.bias": [8],
 "model.model.processor.0.norm.weight": [8],
 "model.model.processor.1.lin_query.base_layer.bias": [8],
 "model.model.processor.1.lin_query.base_layer.weight": [8, 8],
 "model.model.processor.1.lin_query.lora_A.default.weight": [4, 8],
 "model.model.processor.1.lin_query.lora_B.default.weight": [8, 4],
 "model.model.processor.1.lin_value.base_layer.bias": [8],
 "model.model.processor.1.lin_value.base_layer.weight": [8, 8],
 "model.model.processor.1.lin_value.lora_A.default.weight": [4, 8],
 "model.model.processor.1.lin_value.lora_B.default.weight": [8, 4],
 "model.model.processor.1.mlp.0.base_layer.bias": [16],
 "model.model.processor.1.mlp.0.base_layer.weight": [16, 8],
 "model.model.processor.1.mlp.0.lora_A.default.weight": [4, 8],
 "model.model.processor.1.mlp.0.lora_B.default.weight": [16, 4],
 "model.model.processor.1.mlp.2.base_layer.bias": [8],
 "model.model.processor.1.mlp.2.base_layer.weight": [8, 16],
 "model.model.processor.1.mlp.2.lora_A.default.weight": [4, 16],
 "model.model.processor.1.mlp.2.lora_B.default.weight": [8, 4],
 "model.model.processor.1.norm.bias": [8],
 "model.model.processor.1.norm.weight": [8],
 "model.pre_processors.bias": [4],
 "model.pre_processors.weight": [4, 4]}
# fmt: on
