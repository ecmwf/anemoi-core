from collections.abc import Callable
from textwrap import dedent
from typing import Any

import yaml

from anemoi.training.migrations.migrator import MigrationManifest
from anemoi.training.migrations.migrator import Migrator


def test_add(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_add(m: MigrationManifest) -> None:
        m.add("c", 2)

    migrator = migrator_from_funcs(migrate_add, signature="1")
    config = dedent("""\
        a: 0
        b: 1
        version: null
        # test
    """)
    out = migrator.sync(config)
    assert out == dedent("""\
        a: 0
        b: 1
        version: '1'
        # test
        # <<< MIGRATION 1: added key c.
        c: 2
        # >>> MIGRATION 1
    """)


def test_add_nested(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_add(m: MigrationManifest) -> None:
        m.add("c.a", 2)

    migrator = migrator_from_funcs(migrate_add, signature="1")
    config = {"a": 0, "b": 1, "version": None}
    target = {"a": 0, "b": 1, "c": {"a": 2}, "version": "1"}
    out = migrator.sync(yaml.safe_dump(config))
    print(out)
    out = yaml.safe_load(out)
    assert out == target


def test_delete(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_delete(m: MigrationManifest) -> None:
        m.remove("a")

    migrator = migrator_from_funcs(migrate_delete, signature="1")
    config = {"a": 0, "b": 1, "version": None}
    target = {"b": 1, "version": "1"}
    out = yaml.safe_load(migrator.sync(yaml.safe_dump(config)))
    assert out == target


def test_delete_nested(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_delete(m: MigrationManifest) -> None:
        m.remove("a.b")

    migrator = migrator_from_funcs(migrate_delete, signature="1")
    config = {"a": {"a": 0, "b": 1}, "b": 1, "version": None}
    target = {"a": {"a": 0}, "b": 1, "version": "1"}
    out = yaml.safe_load(migrator.sync(yaml.safe_dump(config)))
    assert out == target


def test_move(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.move("a", "b")

    migrator = migrator_from_funcs(migrate_move, signature="1")
    config = {"a": {"b": 0, "c": 1}, "version": None}
    target = {"b": {"b": 0, "c": 1}, "version": "1"}
    out = migrator.sync(yaml.safe_dump(config))
    out = yaml.safe_load(out)
    assert out == target


def test_move_nested(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.move("a.b", "b.b")
        m.move("a.c", "b.c")

    migrator = migrator_from_funcs(migrate_move, signature="1")
    config = {"a": {"b": 0, "c": 1}, "version": None}
    target = {"a": {}, "b": {"b": 0, "c": 1}, "version": "1"}
    out = yaml.safe_load(migrator.sync(yaml.safe_dump(config)))
    assert out == target


def test_move_nested_self(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.move("a", "a.a")

    migrator = migrator_from_funcs(migrate_move, signature="1")
    config = {"a": 0, "version": None}
    target = {"a": {"a": 0}, "version": "1"}
    out = yaml.safe_load(migrator.sync(yaml.safe_dump(config)))
    assert out == target


def test_nest_in_list(migrator_from_funcs: Callable[..., Migrator]):
    def migrate_move(m: MigrationManifest) -> None:
        m.nest_in_list("a")

    migrator = migrator_from_funcs(migrate_move, signature="1")
    config = {"a": {"b": 0, "c": 1}, "version": None}
    target = {"a": [{"b": 0, "c": 1}], "version": "1"}
    out = yaml.safe_load(migrator.sync(yaml.safe_dump(config)))
    assert out == target


def test_transform(migrator_from_funcs: Callable[..., Migrator]):
    def transform_callback(cfg: Any, val: Any) -> Any:
        return val + cfg["b"]

    def migrate_transform(m: MigrationManifest) -> None:
        m.transform("a", transform_callback)

    migrator = migrator_from_funcs(migrate_transform, signature="1")
    config = {"a": 1, "b": 1, "version": None}
    target = {"a": 2, "b": 1, "version": "1"}
    out = yaml.safe_load(migrator.sync(yaml.safe_dump(config)))
    assert out == target
