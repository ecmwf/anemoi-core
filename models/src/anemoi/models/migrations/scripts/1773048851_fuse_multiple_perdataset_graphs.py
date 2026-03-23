# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "0.13.0",
    },
)
# <-- END DO NOT CHANGE


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    dataset_names = list(ckpt["hyper_parameters"]["graph_data"].keys())

    updates = {}
    for old_key in list(ckpt["state_dict"].keys()):
        if not old_key.startswith("model.model.node_attributes."):
            continue

        for dataset_name in dataset_names:
            for key in ["data", "hidden"]:
                # Get old keys from previous checkpoints. The hidden was repeated across all dataset keys.
                old_key_latlons = f"model.model.node_attributes.{dataset_name}.latlons_{key}"
                old_key_trainabletensors = (
                    f"model.model.node_attributes.{dataset_name}.trainable_tensors.{key}.trainable"
                )

                # New checkpoints. Only one `hidden` key now.
                new_key_data_latlons = f"model.model.node_attributes.latlons_{dataset_name}"
                new_key_data_trainabletensors = (
                    f"model.model.node_attributes.trainable_tensors.{dataset_name}.trainable"
                )
                new_key_hidden_latlons = f"model.model.node_attributes.latlons_{key}"
                new_key_hidden_trainabletensors = f"model.model.node_attributes.trainable_tensors.{key}.trainable"

                if old_key == old_key_latlons:
                    new_key = new_key_data_latlons if key == "data" else new_key_hidden_latlons
                    if new_key not in updates:
                        updates[new_key] = ckpt["state_dict"][old_key_latlons]
                        del ckpt["state_dict"][old_key_latlons]

                if old_key == old_key_trainabletensors:
                    new_key = new_key_data_trainabletensors if key == "data" else new_key_hidden_trainabletensors
                    if new_key not in updates:
                        updates[new_key] = ckpt["state_dict"][old_key_trainabletensors]
                        del ckpt["state_dict"][old_key_trainabletensors]

    ckpt["state_dict"].update(updates)

    return ckpt
