# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from textwrap import dedent

from anemoi.training.migrations.interpolations import InterpolationReferences
from anemoi.training.migrations.nodes import NodeContainer
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
    interpolations = InterpolationReferences()
    interpolations.parse_node(node)
    assert interpolations.references == {"baz": {("foo", "bar"), ("other",)}, "other": {("nested",)}}


def test_unknown_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${unknown}
    """)
    node = config_from_content(content)
    interpolations = InterpolationReferences()
    interpolations.parse_node(node)
    assert len(interpolations.references) == 0


def test_multiple_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    other: value 2
    foo:
      bar: this is ${baz} and ${other}
    """)
    node = config_from_content(content)
    interpolations = InterpolationReferences()
    interpolations.parse_node(node)
    assert interpolations.references == {"baz": {("foo", "bar")}, "other": {("foo", "bar")}}


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
    interpolations = InterpolationReferences()
    interpolations.parse_node(node)
    assert interpolations.references == {"baz": {("foo", "bar", 0)}, "other": {("foo", "bar", 1)}}


def test_interpolation_not_root_node(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      other: test
      bar:
        baz: ${foo.other}
    other: value
    inter: ${other}
    """)
    node = config_from_content(content)
    interpolations = InterpolationReferences()
    assert isinstance(node["foo"], NodeContainer)
    interpolations.parse_node(node["foo"])
    assert interpolations.references == {"foo.other": {("foo", "bar", "baz")}}
