# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.models.migrations import MissingMigrationException
from anemoi.models.migrations import migrate_ckpt
from tests.checkpoints.migrations import load_test_migrations


def test_run_all_migrations():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations)
    assert len(done_migrations) == 4
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 4
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_run_last_migration():
    dummy_model = {
        "foo": "foo",
        "migrations": [{"name": "1750840837_add_foo", "versions": {"migration": "1.0.0", "anemoi-models": "0.8.1"}}],
    }

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations)
    assert len(done_migrations) == 3
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 4
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_wrong_order():
    dummy_model = {
        "bar": "bar",
        "migrations": [{"name": "1750841219_add_bar", "versions": {"migration": "1.0.0", "anemoi-models": "0.8.1"}}],
    }

    migrations, _ = load_test_migrations()
    with pytest.raises(MissingMigrationException):
        migrate_ckpt(dummy_model, migrations)


def test_extra_migration():
    dummy_model = {"migrations": [{"name": "dummy", "versions": {"migration": "1.0.0", "anemoi-models": "0.8.1"}}]}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations)
    assert len(done_migrations) == 4
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 5
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_migrate_step():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations, steps=1)
    assert len(done_migrations) == 1
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 1
    assert migrated_model["migrations"][0]["name"] == "1750840837_add_foo"
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" not in migrated_model
    assert "baz" not in migrated_model
    assert "test" not in migrated_model


def test_migrate_no_step():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations, steps=0)
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 0
    assert len(migrated_model) == 1


def test_run_migration_step():
    dummy_model = {
        "foo": "foo",
        "migrations": [{"name": "1750840837_add_foo", "versions": {"migration": "1.0.0", "anemoi-models": "0.8.1"}}],
    }

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations, steps=1)
    assert len(done_migrations) == 1
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 2
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "baz" not in migrated_model
    assert "test" not in migrated_model


def test_rollback_step_from_latest():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, _, _ = migrate_ckpt(dummy_model, migrations)
    rollbacked_model, done_migrations, done_rollbacks = migrate_ckpt(migrated_model, migrations, steps=-1)
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 1
    assert len(rollbacked_model["migrations"]) == 3
    assert "foo" in rollbacked_model and migrated_model["foo"] == "foo"
    assert "bar" in rollbacked_model
    assert "baz" in rollbacked_model
    assert "test" not in rollbacked_model


def test_rollback_step_from_middle():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, _, _ = migrate_ckpt(dummy_model, migrations, steps=2)
    rollbacked_model, done_migrations, done_rollbacks = migrate_ckpt(migrated_model, migrations, steps=-1)
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 1
    assert len(rollbacked_model["migrations"]) == 1
    assert len(rollbacked_model) == 2
    assert "foo" in rollbacked_model


def test_migrate_with_target():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations, target="0.8.1")
    assert len(done_migrations) == 2
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 2


def test_migrate_with_target_latest():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations, target="0.9.0")
    assert len(done_migrations) == 4
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 4


def test_migrate_with_target_future():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, done_migrations, done_rollbacks = migrate_ckpt(dummy_model, migrations, target="0.10.0")
    assert len(done_migrations) == 4
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 4


def test_rollback_with_target():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, _, _ = migrate_ckpt(dummy_model, migrations)
    rollbacked_model, done_migrations, done_rollbacks = migrate_ckpt(migrated_model, migrations, target="0.8.1")
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 2
    assert len(rollbacked_model["migrations"]) == 2


def test_rollback_with_target_same_version():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, _, _ = migrate_ckpt(dummy_model, migrations, target="0.8.1")
    rollbacked_model, done_migrations, done_rollbacks = migrate_ckpt(migrated_model, migrations, target="0.8.1")
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 0
    assert len(rollbacked_model["migrations"]) == 2


def test_no_rollback_with_target_latest():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, _, _ = migrate_ckpt(dummy_model, migrations)
    rollbacked_model, done_migrations, done_rollbacks = migrate_ckpt(migrated_model, migrations, target="0.9.0")
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 0
    assert len(rollbacked_model["migrations"]) == 4


def test_no_rollback_with_target_too_old():
    dummy_model = {}

    migrations, misloaded = load_test_migrations()
    assert len(misloaded) == 0

    migrated_model, _, _ = migrate_ckpt(dummy_model, migrations)
    rollbacked_model, done_migrations, done_rollbacks = migrate_ckpt(migrated_model, migrations, target="0.1.0")
    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 4
    assert len(rollbacked_model["migrations"]) == 0
