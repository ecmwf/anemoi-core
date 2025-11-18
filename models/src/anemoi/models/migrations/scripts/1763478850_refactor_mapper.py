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
    update_layers = ["model.model.{block}.edge_inc", "model.model.{block}.trainable.trainable"]
    for block in ["encoder", "processor", "decoder"]:
        old_block, new_block = block, f"{block}_graph_provider"
        for layer in update_layers:
            ckpt["state_dict"][layer.format(block=new_block)] = ckpt["state_dict"].pop(layer.format(block=old_block))

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
