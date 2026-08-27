# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from textwrap import dedent

from anemoi.training.migrations.testing import ConfigFromContent


def test_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${baz}
    other: ${baz}
    nested: ${other}
    """)
    node = config_from_content(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {("baz",): {("foo", "bar"), ("other",)}, ("other",): {("nested",)}}
    assert interpolations.reverse_refs == {
        ("foo", "bar"): {("baz",)},
        ("other",): {("baz",)},
        ("nested",): {("other",)},
    }


def test_unknown_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${unknown}
    """)
    node = config_from_content(content)
    interpolations = node._interpolation_handler
    assert len(interpolations.references) == 0
    assert len(interpolations.reverse_refs) == 0


def test_multiple_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    other: value 2
    foo:
      bar: this is ${baz} and ${other}
    """)
    node = config_from_content(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {("baz",): {("foo", "bar")}, ("other",): {("foo", "bar")}}
    assert interpolations.reverse_refs == {("foo", "bar"): {("baz",), ("other",)}}


def test_list_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    other: value 2
    foo:
      bar:
        - this is ${baz}
        - this is ${other}
    """)
    node = config_from_content(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {("baz",): {("foo", "bar", 0)}, ("other",): {("foo", "bar", 1)}}
    assert interpolations.reverse_refs == {
        ("foo", "bar", 0): {("baz",)},
        ("foo", "bar", 1): {("other",)},
    }


def test_update_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${baz}
    other: ${baz}
    nested: ${other}
    """)
    node = config_from_content(content)
    interpolations = node._interpolation_handler
    assert interpolations.references == {("baz",): {("foo", "bar"), ("other",)}, ("other",): {("nested",)}}
    assert interpolations.reverse_refs == {
        ("foo", "bar"): {("baz",)},
        ("other",): {("baz",)},
        ("nested",): {("other",)},
    }
    node["foo"].set("bar", "normal str")
    node["nested"] = "${baz}"
    assert interpolations.references == {("baz",): {("other",), ("nested",)}, ("other",): set()}
    assert interpolations.reverse_refs == {
        ("other",): {("baz",)},
        ("nested",): {("baz",)},
    }
    del node["baz"]
    assert interpolations.references == {("baz",): set(), ("other",): set()}
    assert len(interpolations.reverse_refs) == 0
