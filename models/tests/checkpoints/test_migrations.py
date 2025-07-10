# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest

from anemoi.models.migrations import IncompatibleCheckpointException
from anemoi.models.migrations import Migrator
from anemoi.models.migrations import MissingMigrationException


def get_test_migrator() -> Migrator:
    """Load the test migrator with migrations from this folder.

    Returns
    -------
    A Migrator instance
    """
    return Migrator.from_path(Path(__file__).parent / "migrations", "migrations")


def test_run_all_migrations():
    dummy_model = {}

    migrator = get_test_migrator()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(dummy_model)

    assert len(done_migrations) == 4
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 4
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "baz" not in migrated_model
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_run_last_migration():
    dummy_model = {
        "foo": "foo",
        "migrations": [{"name": "1750840837_add_foo", "rollback": lambda x: x}],
    }

    migrator = get_test_migrator()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(dummy_model)

    assert len(done_migrations) == 3
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 4
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_wrong_order():
    dummy_model = {
        "foo": "foo",
        "baz": "baz",
        "migrations": [
            {"name": "1750840837_add_foo", "rollback": lambda x: x},
            {"name": "1750859824_add_baz", "rollback": lambda x: x},
        ],
    }
    migrator = get_test_migrator()

    with pytest.raises(MissingMigrationException):
        migrator.sync(dummy_model)


def test_extra_migration():
    dummy_model = {
        "foo": "foo",
        "migrations": [
            {"name": "1750840837_add_foo", "rollback": lambda x: x},
            {"name": "dummy", "rollback": lambda x: x},
        ],
    }

    migrator = get_test_migrator()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(dummy_model)

    assert len(done_migrations) == 3
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 5
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_migrate_step():
    dummy_model = {}

    migrator = get_test_migrator()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(dummy_model, steps=1)

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

    migrator = get_test_migrator()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(dummy_model, steps=0)

    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 0
    assert len(migrated_model) == 1


def test_run_migration_step():
    dummy_model = {
        "foo": "foo",
        "migrations": [{"name": "1750840837_add_foo", "rollback": lambda x: x}],
    }

    migrator = get_test_migrator()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(dummy_model, steps=1)

    assert len(done_migrations) == 1
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 2
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "baz" not in migrated_model
    assert "test" not in migrated_model


def test_sync_rollback():
    migrator = get_test_migrator()
    dummy_model, _, _ = migrator.sync({})
    # only keep the first two migrations of the first group
    migrator._grouped_migrations[0] = migrator._grouped_migrations[0][:2]

    rollbacked_model, done_migrations, done_rollbacks = migrator.sync(dummy_model)

    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 2
    assert len(rollbacked_model["migrations"]) == 2
    assert "foo" in rollbacked_model and dummy_model["foo"] == "foo"
    assert "bar" in rollbacked_model
    assert "baz" not in rollbacked_model
    assert "test" not in rollbacked_model


def test_rollback_step_from_latest():
    migrator = get_test_migrator()
    dummy_model, _, _ = migrator.sync({})
    rollbacked_model, done_migrations, done_rollbacks = migrator.sync(dummy_model, steps=-1)

    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 1
    assert len(rollbacked_model["migrations"]) == 3
    assert "foo" in rollbacked_model and dummy_model["foo"] == "foo"
    assert "bar" in rollbacked_model
    assert "baz" in rollbacked_model
    assert "test" not in rollbacked_model


def test_rollback_step_from_middle():
    migrator = get_test_migrator()
    dummy_model, _, _ = migrator.sync({}, steps=2)
    rollbacked_model, done_migrations, done_rollbacks = migrator.sync(dummy_model, steps=-1)

    assert len(done_migrations) == 0
    assert len(done_rollbacks) == 1
    assert len(rollbacked_model["migrations"]) == 1
    assert len(rollbacked_model) == 2
    assert "foo" in rollbacked_model


def test_error_migration_past_final():
    migrator = get_test_migrator()
    with pytest.raises(IncompatibleCheckpointException):
        migrator.sync({}, steps=5)


def final_rollback(_):
    raise IncompatibleCheckpointException


def _make_recent_ckpt():
    return {
        "foo": "foo",
        "bar": "bar",
        "test": "baz",
        "migrations": [{"name": "1751895180_final", "rollback": final_rollback}],
    }


def test_migrate_recent_model():
    migrator = get_test_migrator()
    model = _make_recent_ckpt()
    migrated_model, done_migrations, done_rollbacks = migrator.sync(model)

    assert len(done_migrations) == 1
    assert len(done_rollbacks) == 0
    assert len(migrated_model["migrations"]) == 2
    assert migrated_model.get("after", None) == "after"


def test_stop_rollback_to_prev_final():
    migrator = get_test_migrator()
    model, _, _ = migrator.sync(_make_recent_ckpt())
    # only keep the first "final" migration
    migrator._grouped_migrations[-1] = migrator._grouped_migrations[-1][:1]
    assert model.get("after", None) == "after"
    rollbacked_model, _, done_rollbacks = migrator.sync(model)

    assert len(done_rollbacks) == 1
    assert len(rollbacked_model["migrations"]) == 1
    assert "after" not in rollbacked_model


def test_error_rollback_to_old():
    migrator = get_test_migrator()
    model, _, _ = migrator.sync(_make_recent_ckpt())
    with pytest.raises(IncompatibleCheckpointException):
        migrator._rollback(model, steps=2)
