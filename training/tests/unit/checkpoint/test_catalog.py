# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for component catalog with dynamic discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from anemoi.training.checkpoint.catalog import ComponentCatalog
from anemoi.training.checkpoint.exceptions import CheckpointConfigError

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestComponentCatalog:
    """Test ComponentCatalog class with dynamic discovery."""

    def test_list_sources(self) -> None:
        """Test listing available sources."""
        # Since we're using dynamic discovery, the actual sources depend on
        # what's implemented. For now, we just test the method works.
        sources = ComponentCatalog.list_sources()

        assert isinstance(sources, list)
        # The list might be empty if modules don't exist yet
        assert sources == sorted(sources)  # Check it's sorted

    def test_list_loaders(self) -> None:
        """Test listing available loaders."""
        loaders = ComponentCatalog.list_loaders()

        assert isinstance(loaders, list)
        assert loaders == sorted(loaders)  # Check it's sorted

    def test_list_modifiers(self) -> None:
        """Test listing available modifiers."""
        modifiers = ComponentCatalog.list_modifiers()

        assert isinstance(modifiers, list)
        assert modifiers == sorted(modifiers)  # Check it's sorted

    def test_class_to_simple_name(self) -> None:
        """Test converting class names to simple identifiers."""
        # Test various naming patterns
        assert ComponentCatalog._class_to_simple_name("S3Source") == "s3"
        assert ComponentCatalog._class_to_simple_name("LocalSource") == "local"
        assert ComponentCatalog._class_to_simple_name("HTTPSource") == "http"
        assert ComponentCatalog._class_to_simple_name("WeightsOnlyLoader") == "weights_only"
        assert ComponentCatalog._class_to_simple_name("TransferLearningLoader") == "transfer_learning"
        assert ComponentCatalog._class_to_simple_name("FreezeModifier") == "freeze"
        assert ComponentCatalog._class_to_simple_name("LoRAModifier") == "lo_ra"

    def test_discover_components(self) -> None:
        """Test component discovery against real implementations.

        Now that LocalSource and HTTPSource exist, we can test discovery
        against the actual sources module without mocking.
        """
        components = ComponentCatalog._discover_components(
            "anemoi.training.checkpoint.sources",
            "CheckpointSource",
        )

        # Should find real concrete implementations
        assert "local" in components
        assert "http" in components

        # Verify target paths
        assert components["local"] == "anemoi.training.checkpoint.sources.LocalSource"
        assert components["http"] == "anemoi.training.checkpoint.sources.HTTPSource"

        # Should not include the abstract base class
        assert "checkpoint" not in components

    @patch("anemoi.training.checkpoint.catalog.importlib.import_module")
    def test_discover_components_import_error(self, mock_import: MagicMock) -> None:
        """Test that discovery handles import errors gracefully."""
        mock_import.side_effect = ImportError("Module not found")

        # Should return empty dict and not raise
        components = ComponentCatalog._discover_components("non.existent.module", "BaseClass")

        assert components == {}

    def test_discover_components_filters_abstract_classes(self) -> None:
        """Test that discovery filters out abstract base classes.

        The real CheckpointSource base class should NOT appear in
        results, only concrete implementations like LocalSource and
        HTTPSource.
        """
        components = ComponentCatalog._discover_components(
            "anemoi.training.checkpoint.sources",
            "CheckpointSource",
        )

        # Abstract base should not be included
        assert "checkpoint" not in components
        # Only concrete classes should appear
        for name in components:
            # Names come from the class name (RunIdSource -> run_id), not the Hydra group.
            assert name in {"local", "http", "s3", "run_id"}, f"Unexpected component: {name}"

    @patch("anemoi.training.checkpoint.catalog.ComponentCatalog._discover_components")
    def test_get_source_target_when_empty(self, mock_discover: MagicMock) -> None:
        """Test getting source target when no sources are discovered."""
        # Mock discovery to return empty dict
        mock_discover.return_value = {}

        with pytest.raises(CheckpointConfigError, match="Unknown checkpoint source") as exc_info:
            ComponentCatalog.get_source_target("s3")

        assert "Unknown checkpoint source: 's3'" in str(exc_info.value)
        assert "no checkpoint sources are currently available" in str(exc_info.value)

    @patch("anemoi.training.checkpoint.catalog.ComponentCatalog._discover_components")
    def test_get_loader_target_when_empty(self, mock_discover: MagicMock) -> None:
        """Test getting loader target when no loaders are discovered."""
        # Mock discovery to return empty dict
        mock_discover.return_value = {}

        with pytest.raises(CheckpointConfigError, match="Unknown loader strategy") as exc_info:
            ComponentCatalog.get_loader_target("weights_only")

        assert "Unknown loader strategy: 'weights_only'" in str(exc_info.value)
        assert "no loaders are currently available" in str(exc_info.value)

    @patch("anemoi.training.checkpoint.catalog.ComponentCatalog._discover_components")
    def test_get_modifier_target_when_empty(self, mock_discover: MagicMock) -> None:
        """Test getting modifier target when no modifiers are discovered."""
        # Mock discovery to return empty dict
        mock_discover.return_value = {}

        with pytest.raises(CheckpointConfigError, match="Unknown model modifier") as exc_info:
            ComponentCatalog.get_modifier_target("freeze")

        assert "Unknown model modifier: 'freeze'" in str(exc_info.value)
        assert "no modifiers are currently available" in str(exc_info.value)

    def test_discovery_result_is_cached(self) -> None:
        """Discovery runs once. Later calls use the cache."""
        fixed = {
            "s3": "anemoi.training.checkpoint.sources.S3Source",
            "local": "anemoi.training.checkpoint.sources.LocalSource",
        }
        with patch.object(ComponentCatalog, "_discover_components", return_value=fixed) as mock_discover:
            first = ComponentCatalog.list_sources()
            second = ComponentCatalog.list_sources()

        assert mock_discover.call_count == 1
        assert first == second == ["local", "s3"]

    @pytest.mark.parametrize(
        ("getter", "name", "available", "expected"),
        [
            ("get_source_target", "loca", {"local": "x.LocalSource", "http": "x.HTTPSource"}, "Did you mean: local"),
            (
                "get_source_target",
                "locel",
                {"local": "x.LocalSource", "http": "x.HTTPSource"},
                "Available sources: http, local",
            ),
            ("get_loader_target", "weight", {"weights_only": "x.W", "warm_start": "x.S"}, "Did you mean: weights_only"),
        ],
    )
    def test_unknown_name_lists_available_names_and_suggestions(
        self,
        getter: str,
        name: str,
        available: dict[str, str],
        expected: str,
    ) -> None:
        with (
            patch.object(ComponentCatalog, "_discover_components", return_value=available),
            pytest.raises(CheckpointConfigError, match=expected),
        ):
            getattr(ComponentCatalog, getter)(name)
