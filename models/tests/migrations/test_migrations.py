# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
import torch

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import IncompatibleCheckpointException
from anemoi.models.migrations import MigrationOp
from anemoi.models.migrations import Migrator
from anemoi.models.migrations import SaveCkpt


def test_run_all_migrations(old_migrator: Migrator, empty_ckpt: Path):
    _, migrated_model, done_ops = old_migrator.sync(empty_ckpt)

    assert len(done_ops) == 4
    for op in done_ops:
        assert isinstance(op, MigrationOp)
    assert len(migrated_model["migrations"]) == 4
    assert "foo" in migrated_model and migrated_model["foo"] == "foo"
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "baz" not in migrated_model
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def rollback_fn_extra_migration(ckpt: CkptType) -> CkptType:
    """Used in test_extra_migration"""
    return ckpt


def test_extra_migration(old_migrator: Migrator, save_ckpt: SaveCkpt):
    dummy_model = save_ckpt(
        {"foo": "foo"},
        migrations=[{"name": "1750840837_add_foo"}, {"name": "dummy", "rollback": rollback_fn_extra_migration}],
    )

    with pytest.raises(IncompatibleCheckpointException):
        _, _, _ = old_migrator.sync(dummy_model)


def test_break_ckpt_too_old(migrator: Migrator, tmp_path: Path):
    path = tmp_path / "model.ckpt"
    torch.save({"pytorch-lightning_version": "", "migrations": []}, path)
    with pytest.raises(IncompatibleCheckpointException):
        migrator.sync(path)


def test_run_last_migration(old_migrator: Migrator, save_ckpt: SaveCkpt):
    dummy_model = save_ckpt({"foo": "foo"}, migrations=[{"name": "1750840837_add_foo"}])

    _, migrated_model, done_ops = old_migrator.sync(dummy_model)

    assert len(done_ops) == 3
    for op in done_ops:
        assert isinstance(op, MigrationOp)
    assert len(migrated_model["migrations"]) == 4
    assert "bar" in migrated_model and migrated_model["bar"] == "bar"
    assert "test" in migrated_model and migrated_model["test"] == "baz"


def test_migrate_recent_model(migrator: Migrator, recent_ckpt: Path):
    _, migrated_model, done_ops = migrator.sync(recent_ckpt)

    assert len(done_ops) == 1
    assert len(migrated_model["migrations"]) == 2
    assert migrated_model.get("after", None) == "after"
