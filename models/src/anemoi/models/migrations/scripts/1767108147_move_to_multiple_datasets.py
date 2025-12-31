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
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
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
    dummy_dataset_name = "data"

    updates = {}
    for key in list(ckpt["state_dict"].keys()):
        # Update pre-processors
        if key.startswith("model.pre_processors."):
            new_key = key.replace("model.pre_processors.", f"model.pre_processors.{dummy_dataset_name}.")
            updates[new_key] = ckpt["state_dict"][key]
            del ckpt["state_dict"][key]

        # Update post-processors
        if key.startswith("model.post_processors."):
            new_key = key.replace("model.post_processors.", f"model.post_processors.{dummy_dataset_name}.")
            updates[new_key] = ckpt["state_dict"][key]
            del ckpt["state_dict"][key]

        # Update node attributes
        if key.startswith("model.model.node_attributes."):
            new_key = key.replace("model.model.node_attributes.", f"model.model.node_attributes.{dummy_dataset_name}.")
            updates[new_key] = ckpt["state_dict"][key]
            del ckpt["state_dict"][key]

        # Adjust encoder
        if key.startswith("model.model.encoder."):
            new_key = key.replace("model.model.encoder.", f"model.model.encoder.{dummy_dataset_name}.")
            updates[new_key] = ckpt["state_dict"][key]
            del ckpt["state_dict"][key]

        # Adjust decoder
        if key.startswith("model.model.decoder."):
            new_key = key.replace("model.model.decoder.", f"model.model.decoder.{dummy_dataset_name}.")
            updates[new_key] = ckpt["state_dict"][key]
            del ckpt["state_dict"][key]

    ckpt["state_dict"].update(updates)

    ckpt["hyper_parameters"]["data_indices"] = {dummy_dataset_name: ckpt["hyper_parameters"].pop("data_indices")}
    ckpt["hyper_parameters"]["statistics"] = {dummy_dataset_name: ckpt["hyper_parameters"].pop("statistics")}
    ckpt["hyper_parameters"]["statistics_tendencies"] = {
        dummy_dataset_name: ckpt["hyper_parameters"].pop("statistics_tendencies")
    }
    ckpt["hyper_parameters"]["supporting_arrays"] = {
        dummy_dataset_name: ckpt["hyper_parameters"].pop("supporting_arrays")
    }
    return ckpt


def rollback(ckpt: CkptType) -> CkptType:
    """Rollback the checkpoint.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The rollbacked checkpoint dict.
    """
    return ckpt
