# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Removed config keys raise an informative error pointing to the new surface.

``training.run_id`` / ``training.fork_run_id`` / ``system.input.warm_start`` were
removed in favour of ``training.checkpoint.source``. The ``_check_deprecated_keys``
before-validator on :class:`SchemaCommonMixin` rejects them on key *presence* (so a
leftover ``run_id: null`` is still flagged), on both the validated and the
unvalidated config paths.
"""

import pytest
from pydantic import ValidationError

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema


def _nested(path_parts: tuple[str, ...], value: object) -> dict:
    """Build a nested dict ``{a: {b: {c: value}}}`` from dotted path parts."""
    config: dict = {}
    node = config
    for part in path_parts[:-1]:
        node = node.setdefault(part, {})
    node[path_parts[-1]] = value
    return config


@pytest.mark.parametrize("schema_cls", [BaseSchema, UnvalidatedBaseSchema])
@pytest.mark.parametrize(
    ("path_parts", "hint_fragment"),
    [
        (("training", "run_id"), "training.run_id has been removed"),
        (("training", "fork_run_id"), "training.fork_run_id has been removed"),
        (("system", "input", "warm_start"), "system.input.warm_start has been removed"),
    ],
)
def test_removed_run_lineage_key_raises(
    schema_cls: type,
    path_parts: tuple[str, ...],
    hint_fragment: str,
) -> None:
    """A removed key with a value raises a hint naming training.checkpoint.source."""
    config = _nested(path_parts, "some-value")
    with pytest.raises(ValidationError, match=hint_fragment):
        schema_cls(**config)


@pytest.mark.parametrize("schema_cls", [BaseSchema, UnvalidatedBaseSchema])
@pytest.mark.parametrize(
    "path_parts",
    [("training", "run_id"), ("training", "fork_run_id"), ("system", "input", "warm_start")],
)
def test_removed_key_present_as_null_still_raises(schema_cls: type, path_parts: tuple[str, ...]) -> None:
    """The check fires on key presence, so a leftover ``key: null`` is still rejected."""
    config = _nested(path_parts, None)
    with pytest.raises(ValidationError, match="has been removed"):
        schema_cls(**config)
