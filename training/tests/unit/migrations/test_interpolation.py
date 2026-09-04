# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from textwrap import dedent

from anemoi.training.migrations.config import Config
from anemoi.training.migrations.interpolations import Interpolation


def test_interpolation() -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${baz}
    other: ${baz}
    nested: ${other}
    """)
    node = Config(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {
        ("baz",): {Interpolation(("foo", "bar"), "baz"), Interpolation(("other",), "baz")},
        ("other",): {Interpolation(("nested",), "other")},
    }
    assert interpolations.reverse_refs == {
        ("foo", "bar"): {Interpolation(("baz",), "baz")},
        ("other",): {Interpolation(("baz",), "baz")},
        ("nested",): {Interpolation(("other",), "other")},
    }


def test_unknown_interpolation() -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${unknown}
    """)
    node = Config(content)
    interpolations = node._interpolation_handler
    assert len(interpolations.references) == 0
    assert len(interpolations.reverse_refs) == 0


def test_multiple_interpolation() -> None:
    content = dedent("""\
    baz: value
    other: value 2
    foo:
      bar: this is ${baz} and ${other}
    """)
    node = Config(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {
        ("baz",): {Interpolation(("foo", "bar"), "baz")},
        ("other",): {Interpolation(("foo", "bar"), "other")},
    }
    assert interpolations.reverse_refs == {
        ("foo", "bar"): {Interpolation(("baz",), "baz"), Interpolation(("other",), "other")},
    }


def test_list_interpolation() -> None:
    content = dedent("""\
    baz: value
    other: value 2
    foo:
      bar:
        - this is ${baz}
        - this is ${other}
    """)
    node = Config(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {
        ("baz",): {Interpolation(("foo", "bar", 0), "baz")},
        ("other",): {Interpolation(("foo", "bar", 1), "other")},
    }
    assert interpolations.reverse_refs == {
        ("foo", "bar", 0): {Interpolation(("baz",), "baz")},
        ("foo", "bar", 1): {Interpolation(("other",), "other")},
    }


def test_update_interpolation() -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${baz}
    other: ${baz}
    nested: ${other}
    """)
    node = Config(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {
        ("baz",): {Interpolation(("foo", "bar"), "baz"), Interpolation(("other",), "baz")},
        ("other",): {Interpolation(("nested",), "other")},
    }
    assert interpolations.reverse_refs == {
        ("foo", "bar"): {Interpolation(("baz",), "baz")},
        ("other",): {Interpolation(("baz",), "baz")},
        ("nested",): {Interpolation(("other",), "other")},
    }
    node["foo"]["bar"] = "normal str"
    node["nested"] = "${baz}"
    assert interpolations.references == {
        ("baz",): {Interpolation(("other",), "baz"), Interpolation(("nested",), "baz")},
        ("other",): set(),
    }
    assert interpolations.reverse_refs == {
        ("other",): {Interpolation(("baz",), "baz")},
        ("nested",): {Interpolation(("baz",), "baz")},
    }
    del node["baz"]
    assert interpolations.references == {("baz",): set(), ("other",): set()}
    assert len(interpolations.reverse_refs) == 0


def test_relative_interpolation() -> None:
    content = dedent("""\
    x: value
    foo:
      bar:
        baz: ${..x}
      x: test
    """)
    node = Config(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {
        ("foo", "x"): {Interpolation(("foo", "bar", "baz"), "..x")},
    }
    assert interpolations.reverse_refs == {
        ("foo", "bar", "baz"): {Interpolation(("foo", "x"), "..x")},
    }
