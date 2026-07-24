from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from omegaconf import ListConfig

from anemoi.training.migrations.migrator import MigrationManifest
from anemoi.training.migrations.migrator import Migrator


def test_base_add(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_add(m: MigrationManifest) -> None:
        m.add("c", 2)

    migrator = migrator_from_funcs(migrate_add)
    config = {"a": 0, "b": 1, "version": 0}
    target = {"a": 0, "b": 1, "c": 2, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_add_nested(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_add(m: MigrationManifest) -> None:
        m.add("c.a", 2)

    migrator = migrator_from_funcs(migrate_add)
    config = {"a": 0, "b": 1, "version": 0}
    target = {"a": 0, "b": 1, "c": {"a": 2}, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_delete(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_delete(m: MigrationManifest) -> None:
        m.remove("a")

    migrator = migrator_from_funcs(migrate_delete)
    config = {"a": 0, "b": 1, "version": 0}
    target = {"b": 1, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_delete_nested(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_delete(m: MigrationManifest) -> None:
        m.remove("a.b")

    migrator = migrator_from_funcs(migrate_delete)
    config = {"a": {"a": 0, "b": 1}, "b": 1, "version": 0}
    target = {"a": {"a": 0}, "b": 1, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_move(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.move("a", "b")

    migrator = migrator_from_funcs(migrate_move)
    config = {"a": {"b": 0, "c": 1}, "version": 0}
    target = {"b": {"b": 0, "c": 1}, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_move_nested(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.move("a.b", "b.b")
        m.move("a.c", "b.c")

    migrator = migrator_from_funcs(migrate_move)
    config = {"a": {"b": 0, "c": 1}, "version": 0}
    target = {"a": {}, "b": {"b": 0, "c": 1}, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_move_nested_self(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.move("a", "a.a")

    migrator = migrator_from_funcs(migrate_move)
    config = {"a": 0, "version": 0}
    target = {"a": {"a": 0}, "version": 1}
    out = migrator.sync(config)
    assert out == target


def test_base_transform(migrator_from_funcs: Callable[..., Migrator]):
    def transform_callback(cfg: DictConfig | ListConfig, val: Any) -> Any:
        return val + cfg.b

    def migrate_transform(m: MigrationManifest) -> None:
        m.transform("a", transform_callback)

    migrator = migrator_from_funcs(migrate_transform)
    config = {"a": 1, "b": 1, "version": 0}
    target = {"a": 2, "b": 1, "version": 1}
    out = migrator.sync(config)
    assert out == target
