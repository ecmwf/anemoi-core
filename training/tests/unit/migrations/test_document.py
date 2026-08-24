# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from textwrap import dedent

from anemoi.training.migrations.document import Node
from anemoi.training.migrations.testing import DocumentFromContent

HERE = Path(__file__).parent


def test_document(document_from_content: DocumentFromContent):
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    foo_node = document["foo"]
    assert isinstance(foo_node, Node)
    assert document["foo"]["bar"]["baz"].value == "value"


def test_add_comment_around(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document["foo"]["bar"].set_comments(before="<<<", after=">>>")

    expected_output = dedent("""\
    foo:
      # <<<
      bar:
        baz: value
        # >>>
    """)

    assert document.to_yaml() == expected_output


def test_select(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    node = document.select(("foo", "bar"))
    assert node.yaml_node.value == {"baz": "value"}


def test_select_missing(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    node = document.select(("foo", "bar2"), create_missing=True)
    assert node.yaml_node.value == {}


def test_set_comment(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document.set_comments(before="Hello!", after="End!")

    expected_output = dedent("""\
    # Hello!
    foo:
      bar:
        baz: value
    # End!
    """)
    assert document.to_yaml() == expected_output


def test_drop_key(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)

    document = document_from_content(content)
    document.drop_key("foo.bar.old")
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
    """)
    assert document.to_yaml() == expected_output


def test_drop_key_remove_empty(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)

    document = document_from_content(content)
    document.drop_key("foo.bar.old", remove_empty=True)
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
    """)
    assert document.to_yaml() == expected_output


def test_add_key(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document.add_key("foo.bar.new", "new value")
    bar_node = document["foo"]["bar"]
    assert "new" in bar_node
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
        new: new value
    """)
    assert document.to_yaml() == expected_output


def test_add_key_nested_commented(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document.add_key("foo.bar2.baz", "value 2")
    bar_node = document["foo"]["bar2"]
    assert "baz" in bar_node
    bar_node["baz"].set_comments(inline="hello!")
    expected_output = dedent("""\
    foo:
      bar:
        baz: value
      bar2:
        baz: value 2 # hello!
    """)
    assert document.to_yaml() == expected_output


def test_rename_key(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document.rename_key("foo.bar.baz", "foo.bar.new")
    bar_node = document["foo"]["bar"]
    assert "baz" not in bar_node
    assert "new" in bar_node
    expected_output = dedent("""\
    foo:
      bar:
        new: value
    """)
    assert document.to_yaml() == expected_output


def test_rename_no_cleanup(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document.rename_key("foo.bar.baz", "foo.new")
    bar_node = document["foo"]["bar"]
    assert "baz" not in bar_node
    assert "new" in document["foo"]
    expected_output = dedent("""\
    foo:
      bar: {}
      new: value
    """)
    assert document.to_yaml() == expected_output


def test_rename_cleanup(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)

    document = document_from_content(content)
    document.rename_key("foo.bar.baz", "foo.new", remove_empty=True)
    assert "new" in document["foo"]
    expected_output = dedent("""\
    foo:
      new: value
    """)
    assert document.to_yaml() == expected_output


def test_rename_cleanup_list(document_from_content: DocumentFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        - baz: value
    """)

    document = document_from_content(content)
    document.rename_key("foo.bar.0.baz", "foo.new", remove_empty=True)
    assert "new" in document["foo"]
    expected_output = dedent("""\
    foo:
      new: value
    """)
    assert document.to_yaml() == expected_output
