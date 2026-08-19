# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable

import pytest

from anemoi.training.migrations.migrator import Migration
from anemoi.training.migrations.migrator import MigrationManifest
from anemoi.training.migrations.migrator import MigrationMetadata
from anemoi.training.migrations.migrator import Migrator


class MigratorFromFuncs:
    def __call__(self, *funcs: Callable[[MigrationManifest], None], signature: str) -> Migrator:
        migrator = Migrator(
            [
                Migration(
                    name=str(k),
                    metadata=MigrationMetadata(versions={"migration": "1.0.0", "anemoi-training": "0.0.0"}),
                    signature=signature,
                    migrate=func,
                )
                for k, func in enumerate(funcs)
            ],
        )
        return migrator


@pytest.fixture(scope="module")
def migrator_from_funcs() -> MigratorFromFuncs:
    """Load the test migrator with migrations from this folder.

    Returns
    -------
    A Migrator instance
    """
    return MigratorFromFuncs()


@pytest.fixture
def dummy_config():
    return {
        "a": {"a": 0, "b": 1, "c": 2},
    }
