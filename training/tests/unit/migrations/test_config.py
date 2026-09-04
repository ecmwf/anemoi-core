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
from anemoi.training.migrations.nodes import NodeContainer


def test_config() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    foo_node = config["foo"]
    assert isinstance(foo_node, NodeContainer)
    assert isinstance(foo_node["bar"], NodeContainer)
    assert foo_node["bar"]["baz"].value == "value"


def test_add_comment_around() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    assert isinstance(config["foo"], NodeContainer)
    config["foo"]["bar"].set_comments(before="<<<", after=">>>")

    expected_output = dedent("""\
    foo:
      # <<<
      bar:
        baz: value
        # >>>
    """)

    assert config.to_yaml() == expected_output


def test_select() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    node = config.select(("foo", "bar"))
    assert node.yaml_node.value == {"baz": "value"}


def test_select_missing() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    node = config.select(("foo", "bar2"), create_missing=True)
    assert node.yaml_node.value == {}


def test_set_comment() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    config.set_comments(before="Hello!", after="End!")

    expected_output = dedent("""\
    # Hello!
    foo:
      bar:
        baz: value
    # End!
    """)
    assert config.to_yaml() == expected_output


def test_drop_key() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)

    config = Config(content)
    config.drop_key("foo.bar.old")
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
    """)
    assert config.to_yaml() == expected_output


def test_drop_key_remove_empty() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)

    config = Config(content)
    config.drop_key("foo.bar.old", remove_empty=True)
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
    """)
    assert config.to_yaml() == expected_output


def test_add_key() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    config.add_key("foo.bar.new", "new value")
    assert isinstance(config["foo"], NodeContainer)
    bar_node = config["foo"]["bar"]
    assert isinstance(bar_node, NodeContainer)
    assert "new" in bar_node
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
        new: new value
    """)
    assert config.to_yaml() == expected_output


def test_add_key_nested_commented() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    config.add_key("foo.bar2.baz", "value 2")
    assert isinstance(config["foo"], NodeContainer)
    bar_node = config["foo"]["bar2"]
    assert isinstance(bar_node, NodeContainer)
    assert "baz" in bar_node
    bar_node["baz"].set_comments(inline="hello!")
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
      bar2:
        baz: value 2 # hello!
    """)
    assert config.to_yaml() == expected_output


def test_rename_key() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    config.rename_key("foo.bar.baz", "foo.bar.new")
    assert isinstance(config["foo"], NodeContainer)
    bar_node = config["foo"]["bar"]
    assert isinstance(bar_node, NodeContainer)
    assert "baz" not in bar_node
    assert "new" in bar_node
    expected_output = dedent("""\
    foo:
      bar:
        new: value
    """)
    assert config.to_yaml() == expected_output


def test_rename_no_cleanup() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    config.rename_key("foo.bar.baz", "foo.new")
    assert isinstance(config["foo"], NodeContainer)
    bar_node = config["foo"]["bar"]
    assert isinstance(bar_node, NodeContainer)
    assert "baz" not in bar_node
    assert "new" in config["foo"]
    expected_output = dedent("""\
    foo:
      bar: {}
      new: value
    """)
    assert config.to_yaml() == expected_output


def test_rename_cleanup() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = Config(content)
    config.rename_key("foo.bar.baz", "foo.new", remove_empty=True)
    assert isinstance(config["foo"], NodeContainer)
    assert "new" in config["foo"]
    expected_output = dedent("""\
    foo:
      new: value
    """)
    assert config.to_yaml() == expected_output


def test_rename_cleanup_list() -> None:
    content = dedent("""\
    foo:
      bar:
        - baz: value
    """)

    config = Config(content)
    config.rename_key("foo.bar.0.baz", "foo.new", remove_empty=True)
    assert isinstance(config["foo"], NodeContainer)
    assert "new" in config["foo"]
    expected_output = dedent("""\
    foo:
      new: value
    """)
    assert config.to_yaml() == expected_output


def test_rename_interpolation() -> None:
    content = dedent("""\
    x: value
    foo:
      bar:
        baz: ${..x}
      x: test
    other: ${foo.x}
    nested: ${other}
    """)
    config = Config(content)
    config.rename_key("foo.x", "foo.y")
    assert config["foo"]["bar"]["baz"].value == "${foo.y}"
    assert config["other"].value == "${foo.y}"


def test_select_through_interpolation() -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    other:
      key: ${foo}
    """)
    config = Config(content)
    assert config.select("other.key.bar.baz").value == "value"
    assert config.select("other.key.bar.baz").prefix == ("foo", "bar", "baz")


def test_select_through_interpolation_resolving_to_str() -> None:
    content = dedent("""\
    foo: "val"
    other:
      key: "${foo} or ${foo}"
    """)
    config = Config(content)
    assert config.select("other.key").resolved_value == "val or val"
