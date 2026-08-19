# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.migrations.migrator import delete
from anemoi.training.migrations.migrator import select
from anemoi.training.migrations.migrator import update


def test_select():
    cfg = {"a": {"b": 1}, "c": 2}
    assert select(cfg, "a.b") == 1
    assert "a" in cfg
    assert select(cfg, "") == cfg


def test_update_simple():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "a.b", 2)
    assert cfg["a"]["b"] == 2


def test_update_create():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "a.c", 1)
    assert "c" in cfg["a"]
    assert cfg["a"]["c"] == 1


def test_update_create_nested():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "a.c.d", 1)
    assert "c" in cfg["a"]
    assert "d" in cfg["a"]["c"]
    assert cfg["a"]["c"]["d"] == 1


def test_update_replace():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "a", {"c": 2})
    assert cfg["a"] == {"c": 2}


def test_update_list():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "d.1", 0)
    assert cfg["d"][1] == 0
    assert len(cfg["d"]) == 2


def test_update_add_list():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "d.2", 2)
    assert len(cfg["d"]) == 3
    assert cfg["d"][2] == 2


def test_update_list_create():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    update(cfg, "d.2.a", 0)
    assert cfg["d"][2]["a"] == 0


def test_delete():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    delete(cfg, "a")
    assert "a" not in cfg


def test_delete_nested():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    delete(cfg, "a.b")
    assert "b" not in cfg["a"]


def test_delete_list():
    cfg = {"a": {"b": 1}, "c": 2, "d": [0, 1]}
    delete(cfg, "d.0")
    assert len(cfg["d"]) == 1
    assert cfg["d"][0] == 1
