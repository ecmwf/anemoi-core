from collections.abc import Callable

import pytest

from anemoi.training.migrations.migrator import Migration
from anemoi.training.migrations.migrator import MigrationManifest
from anemoi.training.migrations.migrator import MigrationMetadata
from anemoi.training.migrations.migrator import Migrator


class MigratorFromFuncs:
    def __call__(self, *funcs: Callable[[MigrationManifest], None]) -> Migrator:
        migrator = Migrator(
            [
                Migration(
                    name=str(k),
                    metadata=MigrationMetadata(versions={"migration": "1.0.0", "anemoi-training": "0.0.0"}),
                    signature="",
                    migrate=func,
                )
                for k, func in enumerate(funcs)
            ]
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
