# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from textwrap import dedent

from anemoi.training.migrations.testing import ConfigFromContents


def test_config(config_from_contents: ConfigFromContents):
    config = config_from_contents({"config.yaml": dedent("""\
            foo:
              bar:
                baz: value
            """)})
    _docs = config.documents


def test_exec_ops(config_from_contents: ConfigFromContents):
    config = config_from_contents(
        {
            "config.yaml": dedent("""\
            key: val
            foo:
              bar:
                baz: value baz
            prefix:
              foo:
                value: 1
            """),
            "prefix/config.yaml": dedent("""\
            foo:
              bar: value bar
            other: other value
            """),
        }
    )

    config.drop_key("prefix.foo")
    config.exec_ops()
    expected_outputs = {
        "config.yaml": dedent("""\
        key: val
        foo:
          bar:
            baz: value baz
        prefix: {}
        """),
        "prefix/config.yaml": dedent("""\
        other: other value
        """),
    }
    for doc_path, expected_output in expected_outputs.items():
        assert config.documents[doc_path].to_yaml() == expected_output


def test_interpolations(config_from_contents: ConfigFromContents):
    config = config_from_contents(
        {
            "config.yaml": dedent("""\
            key: val
            foo:
              bar:
                baz: ${key}
            prefix:
              foo:
                value: 1
            """),
            "prefix/config.yaml": dedent("""\
            foo:
              bar: ${key}
            other: ${prefix.foo.value}
            """),
        }
    )
    config.parse_interpolations()
    assert config._interpolations.references["key"] == {
        ((), "config.yaml", "foo.bar.baz"),
        (("prefix",), "config.yaml", "foo.bar"),
    }
    assert config._interpolations.references["prefix.foo.value"] == {
        (("prefix",), "config.yaml", "other"),
    }
