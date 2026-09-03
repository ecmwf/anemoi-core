# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path

import pytest

from anemoi.training.migrations.migrator import _CONFIG_MIGRATION_KEY
from anemoi.training.migrations.migrator import ConfigMigration
from anemoi.training.migrations.migrator import ConfigMigrator


@pytest.fixture
def get_migrator() -> Callable[[Iterable[Path]], ConfigMigrator]:
    """Load the test migrator with migrations from this folder.

    Returns
    -------
    A Migrator instance
    """

    def get_sub_migrator(files: Iterable[Path]) -> ConfigMigrator:
        return ConfigMigrator(
            ConfigMigrator._migrations_from_files(
                ConfigMigration,
                files,
                "tests.unit.migrations.migrations",
            ),
            _CONFIG_MIGRATION_KEY,
        )

    return get_sub_migrator
