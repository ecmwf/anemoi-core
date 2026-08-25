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

from anemoi.training.migrations.config import Node
from anemoi.training.migrations.config import NodeList
from anemoi.training.migrations.testing import ConfigFromContent


def test_node(config_from_content: ConfigFromContent):
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)
    node = config_from_content(content)["foo"]
    assert node.yaml_node.value == {"bar": {"baz": "value"}}
    baz_node = node["bar"]["baz"]
    assert isinstance(baz_node, Node)
    assert baz_node.yaml_node.value == "value"
    assert baz_node.cfg == "value"


def test_node_add_item(config_from_content: ConfigFromContent):
    content = dedent("""\
    foo:
      bar:
        baz: value
    """)
    node = config_from_content(content)["foo"]
    node["new"] = "new value"
    assert node["new"].yaml_node.value == "new value"
    assert node["new"].cfg == "new value"


def test_node_del_item(config_from_content: ConfigFromContent):
    content = dedent("""\
    foo:
      bar:
        baz: value
        old: old value
    """)
    node = config_from_content(content)["foo"]["bar"]
    del node["old"]
    with pytest.raises(ValueError):
        _old_val = node["old"]


def test_node_list(config_from_content: ConfigFromContent):
    content = dedent("""\
    foo:
      bar:
        - baz: value 1
        - baz: value 2
    """)
    node = config_from_content(content)["foo"]["bar"]
    assert isinstance(node, NodeList)
    assert node[0].yaml_node.value == {"baz": "value 1"}
    assert node[1].yaml_node.value == {"baz": "value 2"}
    with pytest.raises(ValueError):
        _extra_node = node[2]
    node.append({"baz": "value 3"})
    assert node[2].yaml_node.value == {"baz": "value 3"}
