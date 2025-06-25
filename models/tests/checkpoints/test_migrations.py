import pytest

from anemoi.models.migrations import MissingMigrationException
from anemoi.models.migrations import migrate_ckpt
from tests.checkpoints.migrations import load_test_migrations


def test_run_all_migrations():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    migrated_model, done_migrations = migrate_ckpt(dummy_model, migrations)
    assert len(misloaded) == 0
    assert len(done_migrations) == 2
    assert len(migrated_model["migrations"]) == 2
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"


def test_run_last_migration():
    dummy_model = {"foo": "foo", "migrations": ["1750840837_add_foo"]}

    migrations, misloaded = load_test_migrations()
    migrated_model, done_migrations = migrate_ckpt(dummy_model, migrations)
    assert len(misloaded) == 0
    assert len(done_migrations) == 1
    assert len(migrated_model["migrations"]) == 2
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"


def test_wrong_order():
    dummy_model = {"bar": "bar", "migrations": ["1750841219_add_bar"]}

    migrations, _ = load_test_migrations()
    with pytest.raises(MissingMigrationException):
        migrate_ckpt(dummy_model, migrations)


def test_extra_migration():
    dummy_model = {"migrations": ["dummy"]}

    migrations, misloaded = load_test_migrations()
    migrated_model, done_migrations = migrate_ckpt(dummy_model, migrations)
    assert len(misloaded) == 0
    assert len(done_migrations) == 2
    assert len(migrated_model["migrations"]) == 3
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
