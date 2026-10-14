# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import importlib

import torch

migrate = importlib.import_module(
    "anemoi.models.migrations.scripts.1791980135_drop_stateless_processor_buffers"
).migrate

NORMALIZER_BUFFERS = ("_norm_mul", "_norm_add", "_input_idx", "_output_idx", "_model_output_idx")

# The keys the main-era test checkpoints (2026-08-18) carry and this version no longer has.
REMOVED = [
    *(
        f"model.{side}_processors.data.processors.normalizer.{buffer}"
        for side in ("pre", "post")
        for buffer in NORMALIZER_BUFFERS
    ),
    # tendency processors, and a normaliser under another name in the config
    "model.pre_processors_tendencies.data.processors.normalizer._norm_mul",
    "model.post_processors_tendencies.era5.processors.my_normaliser._norm_add",
    "model.model.node_attributes.latlons_data",
    "model.model.node_attributes.latlons_hidden",
]

KEPT = [
    "model.model.node_attributes.trainable_tensors.data.trainable",
    "model.model.encoder.data.proc.layer_norm_mlp.weight",
    # lookalikes outside the processors and node coordinates
    "model.model.processor.proc.0._norm_mul",
    "model.model.node_attributes.latlons_data.extra",
]


def _checkpoint() -> dict:
    return {"state_dict": {key: torch.ones(2) for key in REMOVED + KEPT}, "hyper_parameters": {}}


def test_drops_the_stateless_buffers_and_keeps_everything_else() -> None:
    ckpt = migrate(_checkpoint())

    assert sorted(ckpt["state_dict"]) == sorted(KEPT)
    assert ckpt["hyper_parameters"] == {}


def test_is_idempotent() -> None:
    once = migrate(_checkpoint())
    twice = migrate(copy.deepcopy(once))

    assert sorted(twice["state_dict"]) == sorted(once["state_dict"])
