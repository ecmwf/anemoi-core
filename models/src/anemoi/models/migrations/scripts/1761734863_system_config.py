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
from anemoi.models.migrations import MigrationContext

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE

def migrate_setup(context: MigrationContext) -> None:
    """
    Migrate setup callback to be run before loading the checkpoint.

    Parameters
    ----------
    context : MigrationContext
       A MigrationContext instance
    """
    context.move_module("aanemoi.training.schemas.hardware", "anemoi.training.schemas.system")

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
