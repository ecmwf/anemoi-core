# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from textwrap import dedent

from anemoi.training.migrations.nodes import NodeContainer
from anemoi.training.migrations.testing import ConfigFromContent


def test_config(config_from_content: ConfigFromContent):
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
    foo_node = config["foo"]
    assert isinstance(foo_node, NodeContainer)
    assert isinstance(foo_node["bar"], NodeContainer)
    assert foo_node["bar"]["baz"].value == "value"


def test_add_comment_around(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
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


def test_select(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
    node = config.select(("foo", "bar"))
    assert node.yaml_node.value == {"baz": "value"}


def test_select_missing(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
    node = config.select(("foo", "bar2"), create_missing=True)
    assert node.yaml_node.value == {}


def test_set_comment(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
    config.set_comments(before="Hello!", after="End!")

    expected_output = dedent("""\
    # Hello!
    foo:
      bar:
        baz: value
    # End!
    """)
    assert config.to_yaml() == expected_output


def test_drop_key(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)

    config = config_from_content(content)
    config.drop_key("foo.bar.old")
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
    """)
    assert config.to_yaml() == expected_output


def test_drop_key_remove_empty(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)

    config = config_from_content(content)
    config.drop_key("foo.bar.old", remove_empty=True)
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
    """)
    assert config.to_yaml() == expected_output


def test_add_key(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
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


def test_add_key_nested_commented(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
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


def test_rename_key(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
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


def test_rename_no_cleanup(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
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


def test_rename_cleanup(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    config = config_from_content(content)
    config.rename_key("foo.bar.baz", "foo.new", remove_empty=True)
    assert isinstance(config["foo"], NodeContainer)
    assert "new" in config["foo"]
    expected_output = dedent("""\
    foo:
      new: value
    """)
    assert config.to_yaml() == expected_output


def test_rename_cleanup_list(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        - baz: value
    """)

    config = config_from_content(content)
    config.rename_key("foo.bar.0.baz", "foo.new", remove_empty=True)
    assert isinstance(config["foo"], NodeContainer)
    assert "new" in config["foo"]
    expected_output = dedent("""\
    foo:
      new: value
    """)
    assert config.to_yaml() == expected_output


def test_rename_interpolation(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    baz: value
    foo:
      bar: ${baz}
    other: ${baz}
    nested: ${other}
    """)
    config = config_from_content(content)
    config.rename_key("baz", "ball")
    assert config["foo"].get("bar").value == "${ball}"
    assert config["other"].value == "${ball}"
    print(config.to_yaml())
