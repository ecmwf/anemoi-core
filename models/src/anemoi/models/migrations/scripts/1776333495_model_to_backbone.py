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
        "anemoi-models": "0.12.0",
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
    old_prefix = "model.model."
    new_prefix = "model.backbone."

    keys_to_rename = [k for k in ckpt["state_dict"] if k.startswith(old_prefix)]
    for old_key in keys_to_rename:
        new_key = new_prefix + old_key[len(old_prefix):]
        ckpt["state_dict"][new_key] = ckpt["state_dict"].pop(old_key)

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
    old_prefix = "model.backbone."
    new_prefix = "model.model."

    keys_to_rename = [k for k in ckpt["state_dict"] if k.startswith(old_prefix)]
    for old_key in keys_to_rename:
        new_key = new_prefix + old_key[len(old_prefix):]
        ckpt["state_dict"][new_key] = ckpt["state_dict"].pop(old_key)

    return ckpt
