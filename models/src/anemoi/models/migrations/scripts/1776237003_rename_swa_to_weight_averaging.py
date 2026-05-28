# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationContext
from anemoi.models.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "0.15.0",
    },
)
# <-- END DO NOT CHANGE


def migrate_setup(context: MigrationContext) -> None:
    """Migrate setup callback to be run before loading the checkpoint.

    Parameters
    ----------
    context : MigrationContext
       A MigrationContext instance
    """
    context.delete_attribute("anemoi.training.schemas.training.SWA")


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    Renames the removed ``training.swa`` config key to ``training.weight_averaging``.
    Any previously-configured SWA is silently disabled; users who want weight
    averaging on a migrated checkpoint should re-enable it via the new
    ``weight_averaging`` Hydra-instantiate config.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    training = ckpt["hyper_parameters"]["config"].training
    training.pop("swa", None)
    training.weight_averaging = None
    return ckpt
