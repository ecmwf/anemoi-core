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
        "anemoi-models": "0.11.0",
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
    num_layers = ckpt["hyper_parameters"]["config"].model.processor.num_layers
    num_chunks = ckpt["hyper_parameters"]["config"].model.processor.num_chunks
    state_dict = ckpt["state_dict"]

    blocks_per_chunk = num_layers // num_chunks
    updates = {}

    for key in [k for k in list(state_dict.keys()) if "processor.proc" in k]:
        parts = key.split(".")
        if not parts[5] == "blocks":  # expecting format model.model.processor.proc.i.blocks.j....
            continue

        chunk_idx = int(parts[4])
        block_idx = int(parts[6])

        flat_idx = chunk_idx * blocks_per_chunk + block_idx
        rest = [""] + parts[7:]
        # reconstruct new key: model.model.processor.proc.<flat_idx>.<rest>
        new_key = "model.model.processor.proc." + str(flat_idx) + ".".join(rest)

        updates[new_key] = state_dict[key]
        del state_dict[key]

    ckpt["state_dict"].update(updates)
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
