# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE


def _dataset_names(ckpt: CkptType) -> list[str]:
    data_indices = ckpt.get("hyper_parameters", {}).get("data_indices")
    if isinstance(data_indices, dict) and data_indices:
        return list(data_indices.keys())
    return ["data"]


def migrate(ckpt: CkptType) -> CkptType:
    """Add GraphProviderRegistry state_dict keys for compatibility."""
    state_dict = ckpt.get("state_dict", {})
    updates = {}

    dataset_names = _dataset_names(ckpt)

    # encoder/decoder providers are per-dataset
    for dataset_name in dataset_names:
        encoder_key = f"model.model.encoder_graph_provider.{dataset_name}.trainable.trainable"
        decoder_key = f"model.model.decoder_graph_provider.{dataset_name}.trainable.trainable"

        encoder_new = f"model.model.graph_providers._providers.{dataset_name}.encoder.trainable.trainable"
        decoder_new = f"model.model.graph_providers._providers.{dataset_name}.decoder.trainable.trainable"
        encoder_top = f"graph_providers._providers.{dataset_name}.encoder.trainable.trainable"
        decoder_top = f"graph_providers._providers.{dataset_name}.decoder.trainable.trainable"

        if encoder_key in state_dict and encoder_new not in state_dict:
            updates[encoder_new] = state_dict[encoder_key]
        if decoder_key in state_dict and decoder_new not in state_dict:
            updates[decoder_new] = state_dict[decoder_key]
        if encoder_key in state_dict and encoder_top not in state_dict:
            updates[encoder_top] = state_dict[encoder_key]
        if decoder_key in state_dict and decoder_top not in state_dict:
            updates[decoder_top] = state_dict[decoder_key]

    # processor provider is stored once (first dataset)
    if dataset_names:
        processor_key = "model.model.processor_graph_provider.trainable.trainable"
        processor_new = f"model.model.graph_providers._providers.{dataset_names[0]}.processor.trainable.trainable"
        processor_top = f"graph_providers._providers.{dataset_names[0]}.processor.trainable.trainable"
        if processor_key in state_dict and processor_new not in state_dict:
            updates[processor_new] = state_dict[processor_key]
        if processor_key in state_dict and processor_top not in state_dict:
            updates[processor_top] = state_dict[processor_key]

    if updates:
        state_dict.update(updates)
        ckpt["state_dict"] = state_dict

    return ckpt


def rollback(ckpt: CkptType) -> CkptType:
    """Rollback is a no-op for this migration."""
    return ckpt
