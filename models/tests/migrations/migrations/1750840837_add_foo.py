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

metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "0.8.1",
    }
)


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint"""
    assert "foo" not in ckpt
    ckpt["foo"] = "foo"
    return ckpt


def rollback(ckpt: CkptType) -> CkptType:
    """Rollbacks the migration"""
    del ckpt["foo"]
    return ckpt
