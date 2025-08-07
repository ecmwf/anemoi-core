# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import IncompatibleCheckpointException
from anemoi.models.migrations import Migrator
from anemoi.models.migrations import OpType


def test_run_all_migrations(old_migrator: Migrator, old_ckpt: CkptType):
    _, migrated_model, done_ops = old_migrator.sync(old_ckpt)

    assert len(done_ops) == 4
    for op_type, _ in done_ops:
        assert op_type is OpType.migration
    assert len(migrated_model["migrations"]) == 4
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "baz" not in migrated_model
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_break_ckpt_too_old(migrator: Migrator):
    with pytest.raises(IncompatibleCheckpointException):
        migrator.sync({})


def test_run_last_migration(old_migrator: Migrator):
    dummy_model = {
        "foo": "foo",
        "migrations": [
            {
                "name": "1750840837_add_foo",
                "rollback": lambda x: x,
                "metadata": {"versions": {"migration": "1.0.0", "anemoi-models": "0.8.0"}},
            }
        ],
    }

    _, migrated_model, done_ops = old_migrator.sync(dummy_model)

    assert len(done_ops) == 3
    for op_type, _ in done_ops:
        assert op_type is OpType.migration
    assert len(migrated_model["migrations"]) == 4
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_extra_migration(old_migrator: Migrator):
    dummy_model = {
        "foo": "foo",
        "migrations": [
            {
                "name": "1750840837_add_foo",
                "rollback": lambda x: x,
                "metadata": {"versions": {"migration": "1.0.0", "anemoi-models": "0.8.1"}},
            },
            {
                "name": "dummy",
                "rollback": lambda x: x,
                "metadata": {"versions": {"migration": "1.0.0", "anemoi-models": "0.9.0"}},
            },
        ],
    }

    _, migrated_model, done_ops = old_migrator.sync(dummy_model)
    assert len(done_ops) == 4
    assert done_ops[0][0] is OpType.rollback
    assert len(migrated_model["migrations"]) == 4
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_migrate_step(old_migrator: Migrator, old_ckpt: CkptType):
    _, migrated_model, done_ops = old_migrator.sync(old_ckpt, steps=1)

    assert len(done_ops) == 1
    assert len(migrated_model["migrations"]) == 1
    assert migrated_model["migrations"][0]["name"] == "1750840837_add_foo"
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" not in migrated_model
    assert "baz" not in migrated_model
    assert "test" not in migrated_model


def test_migrate_no_step(old_migrator: Migrator, old_ckpt: CkptType):
    _, migrated_model, done_ops = old_migrator.sync(old_ckpt, steps=0)

    assert len(done_ops) == 0
    assert len(migrated_model["migrations"]) == 0
    assert len(migrated_model) == 1


def test_run_migration_step(old_migrator: Migrator):
    dummy_model = {
        "foo": "foo",
        "migrations": [
            {
                "name": "1750840837_add_foo",
                "rollback": lambda x: x,
                "metadata": {"versions": {"migration": "1.0.0", "anemoi-models": "0.8.1"}},
            }
        ],
    }

    _, migrated_model, done_ops = old_migrator.sync(dummy_model, steps=1)

    assert len(done_ops) == 1
    assert len(migrated_model["migrations"]) == 2
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "baz" not in migrated_model
    assert "test" not in migrated_model


def test_sync_rollback(old_migrator: Migrator, old_ckpt: CkptType):
    _, dummy_model, _ = old_migrator.sync(old_ckpt)
    # only keep the first two migrations of the first group
    old_grouped_migrations = old_migrator._grouped_migrations[0][:]
    old_migrator._grouped_migrations[0] = old_migrator._grouped_migrations[0][:2]

    _, rollbacked_model, done_ops = old_migrator.sync(dummy_model)

    assert len(done_ops) == 2
    for op_type, _ in done_ops:
        assert op_type is OpType.rollback
    assert len(rollbacked_model["migrations"]) == 2
    assert "foo" in rollbacked_model and dummy_model["foo"] == "foo"
    assert "bar" in rollbacked_model
    assert "baz" not in rollbacked_model
    assert "test" not in rollbacked_model

    # cleanup to not break future tests
    old_migrator._grouped_migrations[0] = old_grouped_migrations


def test_error_migration_past_final(migrator: Migrator, old_ckpt: CkptType):
    with pytest.raises(IncompatibleCheckpointException):
        migrator.sync(old_ckpt, steps=5)


def test_migrate_recent_model(migrator: Migrator, recent_ckpt: CkptType):
    _, migrated_model, done_ops = migrator.sync(recent_ckpt)

    assert len(done_ops) == 1
    assert len(migrated_model["migrations"]) == 2
    assert migrated_model.get("after", None) == "after"


def test_stop_rollback_to_prev_final(migrator: Migrator, recent_ckpt: CkptType):
    _, model, _ = migrator.sync(recent_ckpt)
    # only keep the first "final" migration
    migrator._grouped_migrations[-1] = migrator._grouped_migrations[-1][:1]
    assert model.get("after", None) == "after"
    _, rollbacked_model, done_ops = migrator.sync(model)

    assert len(done_ops) == 1
    assert len(rollbacked_model["migrations"]) == 1
    assert "after" not in rollbacked_model
