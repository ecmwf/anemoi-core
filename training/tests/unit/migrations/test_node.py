# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from textwrap import dedent

import pytest

from anemoi.training.migrations.nodes import Node
from anemoi.training.migrations.nodes import NodeContainer
from anemoi.training.migrations.nodes import NodeList
from anemoi.training.migrations.testing import ConfigFromContent


def test_node(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)
    node = config_from_content(content)["foo"]
    assert isinstance(node, NodeContainer)
    assert node.yaml_node.value == {"bar": {"baz": "value"}}
    assert isinstance(node["bar"], NodeContainer)
    baz_node = node["bar"]["baz"]
    assert isinstance(baz_node, Node)
    assert baz_node.yaml_node.value == "value"
    assert baz_node.cfg == "value"


def test_node_add_item(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)
    node = config_from_content(content)["foo"]
    assert isinstance(node, NodeContainer)
    node["new"] = "new value"
    assert node["new"].yaml_node.value == "new value"
    assert node["new"].cfg == "new value"


def test_node_del_item(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)
    node = config_from_content(content)["foo"]
    assert isinstance(node, NodeContainer)
    bar_node = node["bar"]
    assert isinstance(bar_node, NodeContainer)
    del bar_node["old"]
    with pytest.raises(ValueError, match=r"key old not in Node."):
        _old_val = bar_node["old"]


def test_node_list(config_from_content: ConfigFromContent) -> None:
    content = dedent("""\
    foo:
      bar:
        - baz: value 1
        - baz: value 2
    """)
    node = config_from_content(content)["foo"]
    assert isinstance(node, NodeContainer)
    bar_node = node["bar"]
    assert isinstance(bar_node, NodeList)
    assert bar_node[0].yaml_node.value == {"baz": "value 1"}
    assert bar_node[1].yaml_node.value == {"baz": "value 2"}
    with pytest.raises(ValueError, match=r"key 2 not in Node."):
        _extra_bar_node = bar_node[2]
    bar_node.append({"baz": "value 3"})
    assert bar_node[2].yaml_node.value == {"baz": "value 3"}
