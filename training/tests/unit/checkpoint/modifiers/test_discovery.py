# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator

import pytest

from anemoi.training.checkpoint.base import PipelineStage
from anemoi.training.checkpoint.catalog import ComponentCatalog
from anemoi.training.checkpoint.modifiers import FreezingModifierStage
from anemoi.training.checkpoint.modifiers import ModelModifier


@pytest.fixture(autouse=True)
def _reset_catalog_cache() -> Iterator[None]:
    # ComponentCatalog caches discovery results on the class. Other tests
    # (e.g. test_catalog.py) deliberately populate the cache with mocked empty
    # results, so reset it around each test here to force a fresh discovery.
    ComponentCatalog._sources = None
    ComponentCatalog._loaders = None
    ComponentCatalog._modifiers = None
    yield
    ComponentCatalog._sources = None
    ComponentCatalog._loaders = None
    ComponentCatalog._modifiers = None


def test_modifier_base_extends_pipeline_stage() -> None:
    assert issubclass(ModelModifier, PipelineStage)


def test_freezing_modifier_subclasses_base() -> None:
    assert issubclass(FreezingModifierStage, ModelModifier)


def test_catalog_discovers_freezing_modifier() -> None:
    catalog = ComponentCatalog()
    assert "freezing_modifier_stage" in catalog.list_modifiers()


def test_catalog_resolves_modifier_target() -> None:
    catalog = ComponentCatalog()
    target = catalog.get_modifier_target("freezing_modifier_stage")
    assert target == "anemoi.training.checkpoint.modifiers.FreezingModifierStage"
